from setuptools import find_packages,setup
from typing import List

TRIGGER_VAR = '-e .'

def find_requirements(file_path:str)->List[str]:
    # this function returns list of requirements
    requirements = list()
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if TRIGGER_VAR in requirements:
            requirements.remove(TRIGGER_VAR)
            
    return requirements

setup(
    name='employeeAttritionProject',
    version='0.0.1',
    author='Steve',
    author_email='odhiambostephen057@gmail.com',
    packages=find_packages(),
    install_requires = find_requirements('requirements.txt')
)