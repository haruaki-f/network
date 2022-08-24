import argparse
import cv2

from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER
from paz.backend.image import resize_image, convert_color_space, show_image
from paz.backend.image import BGR2RGB


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()
    print(args.offset, args.camera_id)

    pipeline = DetectMiniXceptionFER([args.offset, args.offset])
    camera = Camera(args.camera_id)
    print("++++++++++++++++++++++++++++++")
    player = VideoPlayer((640, 480), pipeline, camera)
    print("==================================")
    #player.run()

    player.camera.start()
    while True:
        output = player.step()
        print(output["boxes2D"])
        if output is None or len(output["boxes2D"])==0:
            continue
        else:
            emotion=output["boxes2D"][0].class_name
            print(emotion)

        image = resize_image(output[player.topic], tuple(player.image_size))
        show_image(image, 'inference', wait=False)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    player.camera.stop()
    cv2.destroyAllWindows()
    
