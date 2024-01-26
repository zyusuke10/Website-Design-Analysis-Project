import cv2
import os
import sys
import word_recognition
import color_separate_rgb
import DCT
from matplotlib import pyplot as plt
import numpy as np


class LoadImage:
    def __init__(self, file_path):
        self.image = self.read_image(file_path)
    
    def read_image(self, file_path):
        if not os.path.exists(file_path):
            print("The file path doesn't exist")  
            sys.exit() 

        try:
            image = cv2.imread(file_path)
            return image
        
        except cv2.error:
            print("画像読み込みエラーのため、終了します")
            sys.exit()

    @property
    def get_image(self):
        return self.image

class CustomizeImage:
    def __init__(self, image):
        self.customized_image = self.customize(image)

    def customize(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        gauss = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thres = cv2.threshold(gauss, 254, 255, cv2.THRESH_BINARY) 
        return thres
    
class Segmentation:
    def __init__(self, thres,exclusion_regions):
        self.cons = self.segment(thres,exclusion_regions)
    
    def segment(self, thres, exclusion_regions):
        cons, _ = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        filtered_contours = []
        for con in cons:
            if cv2.contourArea(con) < 800:
                continue  
            x, y, w, h = cv2.boundingRect(con)
            skip_contour = False
            for region in exclusion_regions:
                x1, y1, x2, y2 = region
                if x >= x1 and y >= y1 and x + w <= x2 and y + h <= y2:
                    skip_contour = True
                    break
            if not skip_contour:
                filtered_contours.append(con)
        return filtered_contours

class DrawEdges:
    def __init__(self, original_image, cons):
        self.result_image = self.drawEdges(original_image.copy(), cons)
    
    def drawEdges(self, original_image, cons): 
        for con in cons: 
            cv2.polylines(original_image, con, True, (255, 0, 0), 5) 

        return original_image

class Elements:
    def __init__(self):
        self.area_parameter_ratios = []
        self.total_contours = 0
        self.colorQty = 0

    def getTotalContours(self,cons):
        self.total_contours = len(cons)

    
    def getColorQty(self):
        pass


def conductAnalysis(image, country):

    #------Word Recognition ----------------------------------------------------------------
    image_wr = word_recognition.LoadImage(image)
    customized_wr_img = word_recognition.CustomizeImage(image_wr.get_image)
    recognition = word_recognition.WordRecognizer(customized_wr_img.customized_image)
    textQty = recognition.getTextQty(recognition.result)
    exclusion_regions = recognition.get_exclusion_regions(recognition.result)

    # img2 = cv2.imread(image)
    # recognition.showTextPosition(img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#------Segmentation--------------------------------------------------------------------
    image_sg = LoadImage(image)
    customized_sg_img = CustomizeImage(image_sg.get_image)  
    segmentation = Segmentation(customized_sg_img.customized_image,exclusion_regions) 
    #Draw all the edges on the original image
    # result_image = DrawEdges(image_sg.get_image, segmentation.cons).result_image
    # Show result
    # cv2.imshow("Result", result_image)  
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#-----Colour Analysis---------------------------------------------------------------------
    img_loader = color_separate_rgb.SetLoadingImage(image)
    original_image = img_loader.return_img
    k_means = color_separate_rgb.KMeansAnalyzer(original_image)
    df = k_means.analyze()

    color_data = color_separate_rgb.Data(df)
    labs = color_data.rgb2cielab(k_means.rgb_value)
    average_color_similarity = round(color_data.average_similarity(labs),2)

#------Data---------------------------------------------------------------------------
    dct = DCT.LoadImage(image)
    dct_HFC = DCT.DCTCalculator(dct.image)
    high_frequency_coefficients = dct_HFC.high_frequency_coefficients
    scaling_factor = 100000000
    scaled_hfc = round(high_frequency_coefficients / scaling_factor,2)
    
    elements = Elements()
    elements.getTotalContours(segmentation.cons)

    complexity = (elements.total_contours * 0.25) + (textQty * 0.25) + (average_color_similarity * 0.25) + (scaled_hfc * 0.25)

    result = {"country" : country ,"contours" : elements.total_contours, "textQty" : textQty, "averageColorSimilarity" : average_color_similarity, "hfc" : scaled_hfc, "complexity" : complexity}
    return result

def calc_result(websites, title):
    contours = []
    color_similarities = []
    textQty_list = []
    hfc_list = []
    complexity_list = []

    for website in websites:
        result = conductAnalysis(website)
        for key,val in result.items():
            if key == "contours":
                contours.append(val)
            elif key == "textQty":
                textQty_list.append(val)
            elif key == "averageColorSimilarity":
                color_similarities.append(val)
            elif key == "hfc":
                hfc_list.append(val)
            else:
                complexity_list.append(val)

    avg_contours = np.mean(contours)
    avg_color_similarities = np.mean(color_similarities)
    avg_text_qty = np.mean(textQty_list)
    avg_hfc = np.mean(hfc_list)

    contour_weight = 0.25
    color_weight = 0.25
    text_weight = 0.25
    dct_hfc_weight = 0.25

    complexity = (contour_weight * avg_contours) + (color_weight * avg_color_similarities) + (text_weight * avg_text_qty) + (dct_hfc_weight * avg_hfc)

    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    axs = axs.flatten()
    axs[0].scatter(contours, textQty_list)
    axs[0].set_title(title)
    axs[0].set_xlabel('contours')
    axs[0].set_ylabel('textQty')

    axs[1].scatter(contours, color_similarities)
    axs[1].set_title(title)
    axs[1].set_xlabel('contours')
    axs[1].set_ylabel('color similarity')

    axs[2].scatter(contours, hfc_list)
    axs[2].set_title(title)
    axs[2].set_xlabel('contours')
    axs[2].set_ylabel('high frequency coefficients')

    axs[3].scatter(textQty_list, color_similarities)
    axs[3].set_title(title)
    axs[3].set_xlabel('textQty')
    axs[3].set_ylabel('color_similarity')

    axs[4].scatter(textQty_list, hfc_list)
    axs[4].set_title(title)
    axs[4].set_xlabel('textQty')
    axs[4].set_ylabel('high frequency coefficients')

    axs[5].scatter(hfc_list, color_similarities)
    axs[5].set_title(title)
    axs[5].set_xlabel('high frequency coefficients')
    axs[5].set_ylabel('color similarity')

    axs[6].scatter(contours, complexity_list)
    axs[6].set_title(title)
    axs[6].set_xlabel('Contours')
    axs[6].set_ylabel('Complexity')

    axs[7].scatter(textQty_list, complexity_list)
    axs[7].set_title(title)
    axs[7].set_xlabel('textQty')
    axs[7].set_ylabel('Complexity')

    axs[8].scatter(color_similarities, complexity_list)
    axs[8].set_title(title)
    axs[8].set_xlabel('color similarity')
    axs[8].set_ylabel('Complexity')

    axs[9].scatter(hfc_list, complexity_list)
    axs[9].set_title(title)
    axs[9].set_xlabel('high frequency coefficients')
    axs[9].set_ylabel('Complexity')

    plt.tight_layout()
    plt.show()
     
    print(f'Average number of contours : {avg_contours}')
    print(f'Average number of texts : {avg_text_qty}')
    print(f'Average color similarity : {avg_color_similarities}')
    print(f'Average number of High frequency coefficients : {avg_hfc}')
    print(f'Average Design Complexity : {complexity}')


def showDistribution(websites, countries):
    hfc_lists = {country: [] for country in countries}
    for i in range (len(websites)):
        for j in range(len(websites[i])):
            result = conductAnalysis(websites[i][j],countries[i])
            for key, val in result.items():
                if key == "hfc" and result["country"] in hfc_lists:
                     hfc_lists[result["country"]].append(val)
    
    for country, hfc_list in hfc_lists.items():
        plt.scatter([i for i in range(len(hfc_list))], hfc_list, label=country)

    plt.xlabel('Number of High frequency coefficients')
    plt.ylabel('High frequency coefficients in each website out of 100')
    plt.title('High Frequency Coefficients Distribution')
    plt.legend()
    plt.show()



def main():

    dir_path_india = ''
    dir_path_usa = ''
    dir_path_japan = ''

    if os.path.exists(dir_path_japan) and os.path.isdir(dir_path_japan):
        japan = [os.path.join(dir_path_japan, path) for path in os.listdir(dir_path_japan)]
        japan.remove('')
    else:
        print(f"The path {dir_path_japan} does not exist or is not a directory")

    if os.path.exists(dir_path_india) and os.path.isdir(dir_path_india):
        india = [os.path.join(dir_path_india, path) for path in os.listdir(dir_path_india)]
        india.remove('')
    else:
        print(f"The path {dir_path_india} does not exist or is not a directory")

    if os.path.exists(dir_path_usa) and os.path.isdir(dir_path_usa):
        usa = [os.path.join(dir_path_usa, path) for path in os.listdir(dir_path_usa)]
        usa.remove('')
    else:
        print(f"The path {dir_path_usa} does not exist or is not a directory")

    showDistribution([japan,india,usa], ["japan","india","usa"])

    # print(f'/////////////////////////JAPAN RESULT//////////////////////////////////////////')
    # calc_result(japan, "JAPAN")

    # print(f'/////////////////////////INDIA RESULT//////////////////////////////////////////')
    # calc_result(india, "INDIA")

    # print(f'/////////////////////////USA RESULT//////////////////////////////////////////')
    # calc_result(usa, "USA")


if __name__ == "__main__":
    main()

