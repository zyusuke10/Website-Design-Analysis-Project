from PIL import Image,ImageEnhance,ImageFilter
import pyocr
import sys
import os
import cv2

class LoadImage:
    def __init__(self, file_path):
        self.image = self.read_image(file_path)

    def read_image(self, file_path):
        if not os.path.exists(file_path):
            print("The file path doesn't exist")
            sys.exit()

        try:
            if any(file_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                image = Image.open(file_path)
                return image
            else:
                print("The file is not an image")
                return None

        except Exception as e:
            print("Image open error:", e)
            sys.exit()

    @property
    def get_image(self):
        return self.image

class CustomizeImage:
    def __init__(self, image):
        self.customized_image = self.customize(image)

    def customize(self, image):
        grey = image.convert('L')
        filtered_image = grey.filter(ImageFilter.MedianFilter(size=3))
        contrast_enhancer = ImageEnhance.Contrast(filtered_image)
        enhanced_image = contrast_enhancer.enhance(2)
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_sharpness_image = sharpness_enhancer.enhance(5)
        thres = enhanced_sharpness_image.point(lambda p: 255 if p >= 160 else 0)
        thres = self.set_dpi(thres, dpi=300)
        return thres
    
    def set_dpi(self, image, dpi):
        image_copy = image.copy()
        image_copy.info["dpi"] = (dpi, dpi)
        return image_copy 

class WordRecognizer:
    def __init__(self, image):
        self.result = self.recognizer(image)

    def recognizer(self, image):
        try:
            tools = pyocr.get_available_tools()
            tools = tools[0]
            langs = "jpn+eng"
            box_builder = pyocr.builders.WordBoxBuilder(tesseract_layout=4)
            text_position = tools.image_to_string(image,lang=langs,builder = box_builder)  
            return text_position
        except Exception as e:
            print("OCR error:", e)
            sys.exit()
        
    def showTextPosition(self,img2):
        for res in self.result:
            cv2.rectangle(img2,res.position[0],res.position[1],(0,0,255),2)
            cv2.imshow("result1", img2)

    def getTextQty(self,result):
        return len(result)

    
    def get_exclusion_regions(self, result, margin=400):
        exclusion_regions = []
        for box in result:
            tuple1 = box.position[0] 
            tuple2 = box.position[1] 
            
            x1,y1 = tuple1
            x2,y2 = tuple2

            x1 = x1 - margin
            y1 = y1 - margin
            x2 = x2 + margin
            y2 = y2 + margin


            exclusion_regions.append((x1, y1, x2, y2))
        return exclusion_regions
    




