from diffusers import FlowMatchEulerDiscreteScheduler #ts is the flow sampler used in SD3
import torch
from tqdm import tqdm

class FlowEditSampler():
    def __init__(self, pipe): #pipeline is either SD3 or FLUX
        self.pipe = pipe
        self.scheduler = pipe.scheduler #should be the ODE euler solver
        self.device = pipe.device #put everything where the model is put (might break if vram is less?)

        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else:
        #     self.device = pipe.device

        print(f"Using Scheduler: {self.scheduler.__class__.__name__}") #should be euler

    @torch.no_grad() #frozen
    def __call__(self, source_img, source_prompt, target_prompt, neg_prompt_src = "", neg_prompt_tar = "", num_steps = 50, n_min = 0, n_max = 15,
                 cfg_src = 1.5, cfg_target = 5.5):
        '''
        FlowEdit works on the concept of avoiding inversions.
        The authors WANT to draw a line between the source image and the target edit image.

        BUT, this cant be done as flow/diffusion models are trained to predict the score/vector fields from the
        noisy state and NOT go from one image to another.

        To mitigate this, the anuthors need to first take the images to a noisy state. 

        [ NOTE: here, the time convention is:- noise(1) to image(0) or image(0) to noise(1) ]
        'n_max' is the amount of noise added added to the source image. 
        eg- 15/50 = t_{0.3}. So according to the lerp formula: (1 - 0.3)*img + 0.3*noise  
        we also start the process from 'n_max', and perform the flowedit upto 'n_min'. After n_min,
        we do normal conditioning (without vector subtraction).

        'n_min' is until when we want to "see" the source image. by default, this is 0. So, until we form the target(high structural integrity) 
        
        a target_img path is also considered, whose destination will be the final result. At first, this is at hte source image.
        This will follow the vector field to the 'edit'.
        Steps;
        1. src image is taken back to noise(given by n_max as stated above).

        2. The noisy_target_vector is constructed by: (edit_path + noisy_src_img - src_img). At the start,
        because the edit_path is at the src_img, this is just he noisy_src_img.
        In the later timesteps, this essentially becomes the noise added to the source path, conditioned
        on the source prompt. The edit part is what the euler sampler moves over.

        3. Now, the model is fed the noisy_src_img, conditioned on the source prompt.
        Then, it is fed the noisy_target_img, conditioned on the target prompt.

        4. It then predicts the vector fields for both. Because we want to go from src to target, 
        our noise_target_img should move in the direction of the vector field (v_target - v_src)
        SDEdit moves in the direction of v_target, and produces subpar results.

        5. Also notice that v_target is conditioned on the same noise added to v_src.
        So hallucination based on different noises is avoided.

        6. Finally, step the target_img in the direction of (v_target-v_src), until t < n_max and t > n_min 

        7. Repeat for every timestep
        '''
        torch.manual_seed(42) #to replicate flow-edit authors'

        #encode src_img to latents using VAE
        x_src = self.encode_image(source_img)

        #encode the text prompts
        cond_src = self.encode_prompt(source_prompt, neg_prompt_src)
        cond_tgt = self.encode_prompt(target_prompt, neg_prompt_tar)

        #setting the timesteps acc. to n_min, n_max
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        z_fe = x_src.clone() #this is the edit path. intially, same as src latent

        #starting the editing process
        for i,t in tqdm(enumerate(timesteps)):
            remaining_steps = num_steps - i
            if remaining_steps > n_max:
                continue #wait till we reach desired noise level


            t_float = t.item()/1000.0 #because SD3 uses 1000 timestep convention
            dt = -1 / num_steps #each discrete timestep (from 1 -> 0). 
            '''
            this was positive in diffusion models because the convention there seemed to be
            noise(0) and image(1). its the reverse in flow models.
            '''
            #CREATE THE NOISY INPUTS FOR SOURCE LATENT
            noise = torch.randn_like(x_src) #draws from gaussian noise
            z_src_noisy = (1 - t_float) * x_src + t_float * noise # eg- 15/1000 = 0.015

            z_tar_noisy = z_fe + (z_src_noisy - x_src) 

            #get the model predictions of the vector fields
            v_src = self.get_model_output(z_src_noisy, t, cond_src, cfg_src) #conditioned on source prompt
            v_tar = self.get_model_output(z_tar_noisy, t, cond_tgt, cfg_target) #conditioned on target prompt

            if remaining_steps > n_min: #at n_min, we stop using this vector field to guide edit path, and use just v_target
                v_direction = v_tar - v_src
                #update the edit path
                z_fe = z_fe + (v_direction * dt)
            
            else:
                #do normal
                z_fe = z_fe + (v_tar * dt)

        
        return self.decode_latents(z_fe) #decode the edited latent after all these steps


    #HELPER funcs!

    def encode_image(self, image):

        #img is on the CPU, model loaded on GPU
        processed_img = self.pipe.image_processor.preprocess(image)
        processed_img = processed_img.to(self.pipe.vae.device, dtype=self.pipe.vae.dtype) #send to gpu

        with torch.no_grad():
            posterior = self.pipe.vae.encode(processed_img).latent_dist #gives the conditional probability path, based on input image
            latents = posterior.sample() * self.pipe.vae.config.scaling_factor #sampling from this conditional path(?)

        return latents
    
    def encode_prompt(self, prompt, neg_prompt):
        #SD3 has three encoders - clip big, clip small, and T5
        prompts_list = [prompt]

        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt = prompts_list,
            prompt_2 = None, 
            prompt_3 = None,
            negative_prompt = neg_prompt,
            device = self.device,
            do_classifier_free_guidance=True,
        )
    
        final_prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
        final_pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim = 0)

        return final_prompt_embeds, final_pooled_prompt_embeds

    def decode_latents(self, latents):
        latents = latents.to(self.device) #just making sure

        if hasattr(self.pipe.vae.config, "scaling_factor"):
            scaling_factor = self.pipe.vae.config.scaling_factor
        else:
            scaling_factor = self.pipe.vae.config.get("scaling_factor", 1.5305)
        
        #do unscaling (we did this during encoding)
        latents = latents / scaling_factor

        with torch.no_grad():
            image_tensor = self.pipe.vae.decode(latents, return_dict=False)[0] #tuple is returned
        
        image = self.pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

        return image

    def get_model_output(self, latents, t, conditioning, cfg_scale): # 't' is the timestep input to model, acts as signal
        """
        will return the predicted the vector field
        """
        prompt_embeds, pooled_prompt_embeds = conditioning #will get this from encode_prompt() helper
        
        latents_input = torch.cat([latents] * 2) #both unconditioned and conditioned need to be run for CFG
        
        t_input = torch.tensor([t] * 2, device=self.device) #same as above

        
        vf_pred = self.pipe.transformer(
            hidden_states=latents_input,
            timestep=t_input,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]

        v_uncond, v_cond = vf_pred.chunk(2) #'_encode_prompt' prepares the null conditioned text input too, along with caption/prompt
    
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        return v_pred

