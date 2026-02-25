import sys
from pathlib import Path
file_path = Path(__file__).resolve()
path_current = file_path.parents[0]
path_root = file_path.parents[1]

sys.path.append(str(path_root))
sys.path.append(str(path_current))

from generate_environment import generateConstraints, createTemplateInstance, getSceneGraph_data, getSceneGraph_constraint

environment_constraints_dir = '/users/sbsh670/CLEVR-POC/clevr-poc-dataset-gen/data/environments'
templates_path  = '/users/sbsh670/CLEVR-POC/clevr-poc-dataset-gen/image_generation/ConstraintTemplates/constraint_templates.txt'
general_constraints_path = '/users/sbsh670/CLEVR-POC/clevr-poc-dataset-gen/data/general_constraints.txt'



total = 0 ##total objects detected in clevr_sample val
count_geo_none = 0 #number of objects for which geometry_shape could not identify shape
count_agree = 0 #number of objects for which there is agreement between dino and geometry


def nearest_color(rgb):
    rgb = np.array(rgb)
    distances = {k: np.linalg.norm(rgb - v) for k, v in CLEVR_COLORS.items()}
    return min(distances, key=distances.get)

def estimate_size(bbox, image_area):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    ratio = area / image_area
    if ratio > 0.15:
        return "large"
    elif ratio > 0.07:
        return "medium"
    return "small"

def get_region(x, y):
    # Map pixel → approximate CLEVR coordinate space [-5,5]
    x_norm = (x / 480) * 10 - 5
    y_norm = (y / 320) * 10 - 5
    
    for region_id, r in REGIONS.items():
        if r["x"][0] <= x_norm <= r["x"][1] and \
           r["y"][0] <= y_norm <= r["y"][1]:
            return region_id
    return None


def extract_shape_from_phrase(phrase):
    phrase = phrase.lower()
    for s in domain['shape']:
        if s in phrase:
            return s
    return None

def geometry_shape(mask):

    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)

    # Sphere
    if circularity > 0.87:
        return "sphere"

    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    vertices = len(approx)

    # Cone (triangle)
    if vertices == 3:
        return "cone"

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    # Cube (square-ish with 4 strong corners)
    if 0.85 < aspect_ratio < 1.15 and vertices == 4:
        return "cube"

    # Cylinder (elongated OR rounded rectangle)
    if vertices >= 4:
        return "cylinder"

    return None

def fused_shape(phrase, mask):

    dino_shape = extract_shape_from_phrase(phrase)
    geo_shape = geometry_shape(mask)
    total = total + 1
    
    #geo_shape does not return anything - so go with dino
    if geo_shape is None:
        count_geo_none = count_geo_none + 1
        return dino_shape
    
    # DINO confident and geometry agrees
    if dino_shape == geo_shape:
        count_agree= count_agree + 1
        return dino_shape

    # Case 2: DINO exists but geometry differs
    if dino_shape is not None and geo_shape is not None:
        # Trust sphere detection strongly (geometry is very reliable here)
        if geo_shape == "sphere":
            return "sphere"

        # Trust DINO for cube vs cylinder vs cone (text model usually better)
        return dino_shape

    # Case 3: Only one available
    if dino_shape is not None:
        return dino_shape

    if geo_shape is not None:
        return geo_shape

    

def generateEnvironment(num_objects, env_id):
    num_objects = num_objects - 1
    templates_list=[]
    file1 = open(templates_path, 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip()       
        templates_list.append(line)
    templates, negation, across, within  = createTemplateInstance(templates_list)
    
    file1.close()
    file1 = open(general_constraints_path, 'r')
    Lines = file1.readlines()
    background = ""
    for line in Lines:
        background = background+line
    file1.close()
    background = background+"\n"+"object(0.."+str(num_objects)+")."+"\n"    
    satisfiable = False
    while(not(satisfiable)):
        asp_file = open(os.path.join(environment_constraints_dir, str(env_id)+".lp"), "w")
        constraints = generateConstraints(templates, negation, across, within)
        asp_code = background+constraints+"\n"+"#show hasProperty/3. #show at/2."
        n1 = asp_file.write(asp_code)
        asp_file.close()
        asp_command = 'clingo 1'  + ' ' + os.path.join(environment_constraints_dir, str(env_id)+".lp")
        output_stream = os.popen(asp_command)
        output = output_stream.read()
        output_stream.close()
        answers = output.split('Answer:')
        #print("Answers:", answers)
        answers = answers[1:]
        count = 0
        for answer in answers:
            count = count+1
            if(count>=1):
                satisfiable = True
                print("Satisfiable")
                break


       

def main(args):
    
   import kagglehub
   # Specify the folder where you want the dataset
   download_path = "/users/sbsh670/archive/clevr"  # <-- your desired path
   path = kagglehub.dataset_download("timoboz/clevr-dataset", path=download_path)
   print("Path to dataset files:", path)
    #env_id = 0
    #num_objects = [5,6,7,8,9]
    #generateEnvironment(num_objects[0], env_id)

            
if __name__ == "__main__":
    main(None)
