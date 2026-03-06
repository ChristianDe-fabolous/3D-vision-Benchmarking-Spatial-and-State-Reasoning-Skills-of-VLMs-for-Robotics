# 3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics


google drive link: https://drive.google.com/drive/folders/1gMF-vDXdjZAspC9u8j9JzT4NB8wk7aGd




## Chris's summary paper:
1. Executive summary
VLMs have recently 
In this project, we will analyze the performance of vision-language models~(VLMs) in the context of robootics, specifically focusing on failure modes and multiview consistency. We will use a benchmark dataset that includes vairous robotic tasks and scenarios to evaluate the VLMs' capabilities in spatial and state reasoning. Our goal is to identify the strengths and weaknesses of current VLMs and provide insights for future improvements in their application to robotics.


2. Problem Statement
* VLMs have oto understand current stages correctly especially if a task failed


3. Project Goal and Objectives
We investigate two scenarios: 
    1. Robotic failure modes.  As well as if they can analyze a state to detect the current mode of a task. 
    2. Multi-view Consistency: For that we research if VLA's can extract relevant spatial information if given multiple perspectives. Can the VLM's correctly detet ... and does multi-perspective increase effectivness.


Benchmark Qwen (maybe more, such as pi, RT2)

4.  Methodology and Approach
* Seperation of tasks
* Dataset understanding: 
    1. How to use data for answering hypothesis
    2. Sorting and cleaning data
    3. Labeling data for specfic questions 
    4. Augmenting data for additional 

    2 and 3 are connected to pipeline building

* Model set up, pipeline building and introduction for others
    1. logging (very important)
    2. modular pipeline
    3. maintainable code
    4. cluster introduction
    5. which format will data be

* Writing and organization
    * slides for each meeting
    * writing project advancements and text
    * literature research
      

5. Timeline and Milestoes





## Dataset
* Wrong answer #5 #35 #37 #67 #69

### Failure modes
specific questions exist
reverse order of images as if robots has done the opposite


### Multiview modes
take 2 camera angles, one with space points. Reason which is closest to other camera (manual task) - similar questions exist
multi view images exist


## Open questions:
* Do we train ourselves?
* We can manipulate and augment the dataset if needed, right?