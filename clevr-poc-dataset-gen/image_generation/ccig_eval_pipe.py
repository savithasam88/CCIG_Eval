import random
import copy, os
import sys, math

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


def evaluate(image, constraints):
       

def main(args):
    env_id = 0
    num_objects = [5,6,7,8,9]
    generateEnvironment(num_objects[0], env_id)

            
if __name__ == "__main__":
    main(None)
