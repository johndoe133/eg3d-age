import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from PIL import Image


def denormalize(z, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
    """Cant import from training.training_loop due to naming of folders...
    """
    x = (z*(rmax - rmin)- tmin*(rmax-rmin))/(tmax-tmin)+rmin
    return x

class SingleIDCoach(BaseCoach):

    def __init__(self, c, image_name, data_loader, use_wandb, evaluation=False, trunc=1.0):
        self.c = c
        self.image_name = image_name
        self.evaluation = evaluation
        self.trunc = trunc
        super().__init__(data_loader, use_wandb)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        # os.makedirs(w_path_dir, exist_ok=True)
        # os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        for fname, image in (self.data_loader):
            if (fname[0] != self.image_name): 
                continue # dont train on the images that is not specified in shell script pti.sh
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{paths_config.embedding_base_dir}/w' #/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                z_pivot, age_pivot = self.calc_inversions(image, image_name, self.c, self.trunc)
                age_pivot = age_pivot.detach()
                self.c[:,-1] = age_pivot # update the conditioning parameters so that the found
                # optmized age is used instead of the previously estimated starting point
                print("Optimized age:", denormalize(age_pivot, rmin=hyperparameters.age_min, rmax=hyperparameters.age_max).item())


            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            z_pivot = z_pivot.to(global_config.device)
            torch.save(z_pivot, f'{embedding_dir}/{image_name}.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            images = []
            for i in tqdm(range(hyperparameters.max_pti_steps)):
                
                generated_images = self.forward(z_pivot, self.trunc)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, z_pivot)
                # save progress
                if i%(hyperparameters.max_pti_steps//6)==0:
                    save_img = (generated_images.permute(0, 2, 3, 1)* 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    images.append(save_img)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    save_img = (generated_images.permute(0, 2, 3, 1)* 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    images.append(save_img)
                    return age_pivot
                    

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name]) # not implemented

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1
            model_path = f'{paths_config.checkpoints_dir}/G'
            os.makedirs(model_path, exist_ok=True)
            model_name = os.path.join(model_path, f'{image_name}.pt')
            torch.save(self.G, model_name)
            print("Saved G at", model_name)

            # save image
            home_dir = os.path.expanduser('~')
            path = f"Documents/eg3d-age/eg3d/PTI/output/{image_name}"
            save_name = os.path.join(home_dir, path, "pti_optimization.png")
            img = torch.cat(images, dim=2)
            Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_name)
            return age_pivot