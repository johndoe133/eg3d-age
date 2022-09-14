from utils.alignment import align_face
import dlib 
import os

def pre_process_images(
    input_image_path = "./PTI/image_original", 
    output_image_path = "./PTI/image_processed",
    predictor_path="./networks/align.dat",
    IMAGE_SIZE = 512):

    print("Running preprocess on images...")
    predictor = dlib.shape_predictor(predictor_path)
    print(os.getcwd())
    aligned_images = []
    print("Aligning faces...")
    image_names = os.listdir(input_image_path)
    for image_name in image_names:
        aligned_face = align_face(filepath=os.path.join(input_image_path, image_name),
                                        predictor=predictor, output_size=IMAGE_SIZE)
        aligned_images.append(aligned_face)
    
    print("Saving aligned faces...")

    for image, name in zip(aligned_images, image_names):
        real_name = name.split('.')[0]
        image.save(f'{output_image_path}/{real_name}.jpeg')
    print("Ending...")

if __name__ == "__main__":
    pre_process_images()