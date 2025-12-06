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
        
        # Helper check to ensure correct device usage even with offloading
        if torch.cuda.is_available():
             self.device = torch.device("cuda")
        else:
             self.device = pipe.device

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

        2. the noisy_target_vector is constructed by: (edit_path + noisy_src_img - src_img). At the start,
        because the edit_path is at the src_img, this is just he noisy_src_img.
        In the later timesteps, this essentially becomes the noise added to the source path, conditioned
        on the source prompt. The edit part is what the euler sampler moves over.

        3. now, the model is fed the noisy_src_img, conditioned on the source prompt.
        Then, it is fed the noisy_target_img, conditioned on the target prompt.

        4. it then predicts the vector fields for both. Because we want to go from src to target, 
        our noise_target_img should move in the direction of the vector field (v_target - v_src)
        SDEdit moves in the direction of v_target, and produces subpar results.

        5. also notice that v_target is conditioned on the same noise added to v_src.
        So hallucination based on different noises is avoided.

        6. finally, step the target_img in the direction of (v_target-v_src), until t < n_max and t > n_min 

        7. repeat for every timestep
        '''
        # torch.manual_seed(42) # REMOVED: Seed is now handled globally in process_dataset.py

        #encode src_img to latents using VAE
        x_src = self.encode_image(source_img)

        #encode the text prompts
        # FIX: Batch encoding logic to support separate negative prompts and batched forward pass
        src_prompt_embeds, src_neg_prompt_embeds, src_pooled_embeds, src_neg_pooled_embeds = self.pipe.encode_prompt(
            prompt=source_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=neg_prompt_src,
            device=self.device,
            do_classifier_free_guidance=True,
        )
        
        tar_prompt_embeds, tar_neg_prompt_embeds, tar_pooled_embeds, tar_neg_pooled_embeds = self.pipe.encode_prompt(
            prompt=target_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=neg_prompt_tar,
            device=self.device,
            do_classifier_free_guidance=True,
        )
        
        # Concatenate for batched processing: [src_uncond, src_cond, tar_uncond, tar_cond]
        src_tar_prompt_embeds = torch.cat([
            src_neg_prompt_embeds, 
            src_prompt_embeds, 
            tar_neg_prompt_embeds, 
            tar_prompt_embeds
        ], dim=0)
        
        src_tar_pooled_embeds = torch.cat([
            src_neg_pooled_embeds,
            src_pooled_embeds,
            tar_neg_pooled_embeds,
            tar_pooled_embeds
        ], dim=0)

        #setting the timesteps acc. to n_min, n_max
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas

        z_fe = x_src.clone() #this is the edit path. intially, same as src latent

        #starting the editing process
        for i,t in tqdm(enumerate(timesteps)):
            remaining_steps = num_steps - i
            if remaining_steps > n_max:
                continue #wait till we reach desired noise level


            # FIX: Use linear math to match official repo (t/1000) instead of scheduler sigmas
            t_curr = t.item() / 1000.0 
            
            # Calculate dt
            if i + 1 < len(timesteps):
                t_next = timesteps[i + 1].item() / 1000.0
            else:
                t_next = 0.0
            dt = t_next - t_curr

            if remaining_steps > n_min: #at n_min, we stop using this vector field to guide edit path, and use just v_target
                # FlowEdit phase with BATCHED computation
                noise = torch.randn_like(x_src).to(self.device)
                
                # Using linear t_curr for mixing matches the official code
                z_src_noisy = (1 - t_curr) * x_src + t_curr * noise
                z_tar_noisy = z_fe + (z_src_noisy - x_src)
                
                # Batch all 4 inputs together [src_uncond, src_cond, tar_uncond, tar_cond]
                batched_latents = torch.cat([z_src_noisy, z_src_noisy, z_tar_noisy, z_tar_noisy])
                
                # Single batched forward pass
                v_src, v_tar = self.get_model_output_batched(
                    batched_latents, 
                    t, 
                    src_tar_prompt_embeds, 
                    src_tar_pooled_embeds,
                    cfg_src,
                    cfg_target
                )
                
                v_direction = v_tar - v_src
                
                #update the edit path
                z_fe = z_fe.to(torch.float32)
                z_fe = z_fe + dt * v_direction
                z_fe = z_fe.to(v_direction.dtype)
            
            else:
                #do normal (SDEdit phase)
                if remaining_steps == n_min:
                    noise = torch.randn_like(x_src).to(self.device)
                    # For the switch, Official code uses scale_noise (which uses sigma)
                    # We replicate that logic here
                    self.scheduler._init_step_index(t)
                    sigma = self.scheduler.sigmas[self.scheduler.step_index]
                    xt_src = sigma * noise + (1.0 - sigma) * x_src
                    xt_tar = z_fe + (xt_src - x_src)
                    z_fe = xt_tar
                
                # For SDEdit, batch xt_tar 4 times like official code
                batched_latents = torch.cat([z_fe, z_fe, z_fe, z_fe])
                
                _, v_tar = self.get_model_output_batched(
                    batched_latents,
                    t,
                    src_tar_prompt_embeds,
                    src_tar_pooled_embeds,
                    cfg_src,
                    cfg_target
                )

                z_fe = z_fe.to(torch.float32)
                z_fe = z_fe + dt * v_tar
                z_fe = z_fe.to(v_tar.dtype)

        
        return self.decode_latents(z_fe) #decode the edited latent after all these steps


    #HELPER funcs!

    def encode_image(self, image):
        dtype = self.pipe.vae.dtype
        vae_device = self.pipe.vae.device

        #image encoding using sd3 VAE
        # Send input to where the VAE is (CPU or GPU)
        image = self.pipe.image_processor.preprocess(image).to(vae_device, dtype=dtype)

        if hasattr(self.pipe.vae, "disable_slicing"):
            self.pipe.vae.disable_slicing()
        
        posterior = self.pipe.vae.encode(image).latent_dist #mean and variance

        latents = posterior.sample()

        if hasattr(self.pipe.vae.config, "shift_factor") and self.pipe.vae.config.shift_factor is not None:
            shift_factor = self.pipe.vae.config.shift_factor
        else:
            shift_factor = 0.0 #
            
        if hasattr(self.pipe.vae.config, "scaling_factor"):
            scaling_factor = self.pipe.vae.config.scaling_factor
        else:
            scaling_factor = 1.5305

        # APPLY SHIFT THEN SCALE!! (if u dont do ts it wont replicate!!)
        latents = (latents - shift_factor) * scaling_factor

        return latents.to(device=self.device, dtype=self.pipe.transformer.dtype)
    
    def decode_latents(self, latents):
        # Move latents to VAE device
        vae_device = self.pipe.vae.device
        latents = latents.to(vae_device)

        if hasattr(self.pipe.vae.config, "shift_factor") and self.pipe.vae.config.shift_factor is not None:
            shift_factor = self.pipe.vae.config.shift_factor
        else:
            shift_factor = 0.0

        if hasattr(self.pipe.vae.config, "scaling_factor"):
            scaling_factor = self.pipe.vae.config.scaling_factor
        else:
            scaling_factor = self.pipe.vae.config.get("scaling_factor", 1.5305)

        #do unscaling (we did this during encoding) and shifting
        latents = (latents / scaling_factor) + shift_factor

        latents = latents.to(self.pipe.vae.device, dtype=self.pipe.vae.dtype)

        with torch.no_grad():
            image_tensor = self.pipe.vae.decode(latents, return_dict=False)[0] #tuple is returned
        
        image = self.pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

        return image

    def get_model_output_batched(self, batched_latents, t, prompt_embeds, pooled_embeds, 
                                  cfg_src, cfg_tar):
        """
        Batched computation matching official implementation
        Input: [src_uncond, src_cond, tar_uncond, tar_cond]
        Output: v_src, v_tar (both with CFG applied)
        """
        # Ensure transformer is on GPU
        if hasattr(self.pipe.transformer, "device") and self.pipe.transformer.device.type != "cuda":
             self.pipe.transformer.to(self.device)

        t_input = t.expand(batched_latents.shape[0])
        
        with torch.no_grad():
            vf_pred = self.pipe.transformer(
                hidden_states=batched_latents,
                timestep=t_input,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False
            )[0]
        
        # Chunk into 4 parts
        src_uncond, src_cond, tar_uncond, tar_cond = vf_pred.chunk(4)
        
        # Apply CFG separately for source and target
        v_src = src_uncond + cfg_src * (src_cond - src_uncond)
        v_tar = tar_uncond + cfg_tar * (tar_cond - tar_uncond)
        
        return v_src, v_tar