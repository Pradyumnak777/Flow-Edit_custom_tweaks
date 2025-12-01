from diffusers import FlowMatchEulerDiscreteScheduler #ts is the flow sampler used in SD3
import torch
from tqdm import tqdm

class FlowEditSampler():
    def __init__(self, pipe): #pipeline is either SD3 or FLUX
        self.pipe = pipe
        self.scheduler = pipe.scheduler
        self.device = pipe.device

        print(f"Using Scheduler: {self.scheduler.__class__.__name__}") #should be euler


    def __call__(self, source_img, source_prompt, target_prompt, num_steps = 50, n_min = 0, n_max = 15,
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

        'n_min' is until when we want to "see" the source image. by default, this is 0. So, until we form the target(high structural integrity) 
        
        a target_img path is also considered, whose destination will be the final result. At first, this is at hte source image.
        This will follow the vector field to the 'edit'.
        Steps;
        1. src image is taken back to noise(given by n_max as stated above).

        2. The noisy_target_vector is constructed by: (edit_path + noisy_src_img - src_img). At the start,
        because the edit_path is at the src_img, this is just he noisy_src_img.
        In the later timesteps, this essentially becomes the noise added to the source path, conditioned
        on the source prompt.

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

        
        device = source_img.device
        dtype = source_img.dtype

        #encode src_img to latents using VAE
        x_src = self.encode_image(source_image)

        #encode the text prompts
        cond_src = self.encode_prompt(source_prompt)
        cond_tgt = self.encode_prompt(target_prompt)

        #setting the timesteps acc. to n_min, n_max
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        z_fe = x_src.clone() #this is the edit path. intially, same as src latent

        #starting the editing process
        for i,t in tqdm(enumerate(timesteps)):
            remaining_steps = num_steps - i
            if remaining_steps > n_max:
                continue #wait till we reach desired noise level


            t_float = t.item()/1000.0
            dt = -1 / num_steps #each discrete timestep (from 1 -> 0)

            #CREATE THE NOISY INPUTS FOR SOURCE LATENT
            noise = torch.randn_like(x_src)
            z_src_noisy = (1 - t_float) * x_src + t_float * noise

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


            



