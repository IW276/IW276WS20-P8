# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import csv
from shutil import copy2 as copy
import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Opening JSON file
    with open('valid_coco.json') as json_file, \
            open('validation-annotations-bbox.csv', 'w', newline='') as train_output:
            #, open('validation-annotations-bbox.csv', 'w', newline='') as validation_output:
        data = json.load(json_file)
        writer = csv.writer(train_output)
        #validation_writer = csv.writer(validation_output)

        #create dict with image ids
        metadata_of_images = dict()
        for image in data["images"]:
            metadata_of_images[image["id"]] = [image["file_name"], image["height"], image["width"]]

        #create csv
        rows = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside", "ClassName"]
        writer.writerow(rows)
        #validation_writer.writerow(rows)
        for annotation in data["annotations"]:
            if "bbox" in annotation.keys():
                box = annotation["bbox"]
                image_metadata = metadata_of_images[annotation["image_id"]]
                height = image_metadata[1]
                width = image_metadata[2]
                writer.writerow([
                                 image_metadata[0][:-4],
                                 "xclick",
                                 "/m/083mg",
                                 1,
                                 float(box[0])/float(width),
                                 float(box[0]+box[2])/float(width),
                                 float(box[1])/float(height),
                                 float(box[1]+box[3])/float(height),
                                  0, 0, 0, 0, 0,
                                 "Person"])

    dir = os.path.dirname(os.path.realpath(__file__))
   # copy(os.path.join(dir, 'train-annotations-bbox.csv'), os.path.join(dir, 'sub-train-annotations-bbox.csv'))
    copy(os.path.join(dir, 'validation-annotations-bbox.csv'), os.path.join(dir, 'sub-test-annotations-bbox.csv'))
    copy(os.path.join(dir, 'validation-annotations-bbox.csv'), os.path.join(dir, 'test-annotations-bbox.csv'))
    copy(os.path.join(dir, 'validation-annotations-bbox.csv'), os.path.join(dir, 'sub-validation-annotations-bbox.csv'))
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
