import click
import os
import cv2
from process import solve_maze
from predict import predict_the_class

def rotate_and_save_image(input_path, direction='right', times=1,
                          output_dir=os.path.join('Slike', "rotated")):
    """
    Funkcija rotira sliku u zadatom pravcu (lijevo ili desno) za 90 stepeni više puta
    i snima rezultat u izlazni direktorijum sa izmijenjenim imenom.

    :param input_path: putanja do originalne slike
    :param direction: pravac rotacije ('left' ili 'right')
    :param times: broj rotacija po 90 stepeni
    :param output_dir: folder u koji se snima rezultat
    :return: putanja do snimljene, rotirane slike
    """
    # Load the image from disk
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot load image: {input_path}")

    # Normalize rotation count
    times = times % 4
    if direction == 'left':
        times = (4 - times) % 4  # Convert to equivalent number of clockwise rotations

    # Apply rotation(s)
    for _ in range(times):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build the output filename
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    new_name = f"{name}_rotated_{direction}_{times}{ext}"
    output_path = os.path.join(output_dir, new_name)

    # Save the rotated image
    cv2.imwrite(output_path, img)
    return output_path

@click.command()
@click.argument('image_path', required=True )
def main(image_path):
    # image_path = rotate_and_save_image(image_path) # pošto je kamera pogrešno zalijepljena :D
    predicted_class = predict_the_class(image_path)
    # ovdje može da ide ručni posao:
    #  - ukoliko je slika dobro predviđena - ubaciti je u odgovarajući train folder
    #  - prije neke sljedeće predikcije - istrenirati!
    #  - a i ne mora
    image_name = os.path.join("original_lavirint",f"slika{predicted_class + 1}.png")
    path = solve_maze(image_name)
    print(f"Path for {image_name}:", path)

if __name__ == "__main__":
    main()