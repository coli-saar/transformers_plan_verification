from setuptools import setup

with open('requirements.txt', 'r', encoding='utf-8') as r:
    requirements = [line.strip() for line in r]

setup(name='llm_plan_verification_full',
      packages=[
          'data_generation',
          'planning_utils',
          'training'
      ],
      install_requires=requirements
      )
