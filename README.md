# 3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics


google drive link: https://drive.google.com/drive/folders/1gMF-vDXdjZAspC9u8j9JzT4NB8wk7aGd




## Chris's summary paper:
1. Executive summary
VLMs have recently 
In this project, we will analyze the performance of vision-language models~(VLMs) in the context of robootics, specifically focusing on failure modes and multiview consistency. We will use a benchmark dataset that includes vairous robotic tasks and scenarios to evaluate the VLMs' capabilities in spatial and state reasoning. Our goal is to identify the strengths and weaknesses of current VLMs and provide insights for future improvements in their application to robotics.


2. Problem Statement
Frontier VLMs can reason over visual inputs derive follow task instructions. They are often trained through 
* VLMs have to understand current stages correctly especially if a task failed


3. Project Goal and Objectives
We investigate two scenarios: 
    1. Robotic failure modes.  As well as if they can analyze a state to detect the current mode of a task.
    2. Multi-view Consistency: For that we research if VLA's can extract relevant spatial information if given multiple perspectives. Can the VLM's correctly detet ... and does multi-perspective increase effectivness.
    In both cases the VLM has to correcly identify and output the answer from multiple in context answers. The visual annotations are created through editing scene images. 

Benchmark Qwen (maybe more, such as pi, RT2)

4.  Methodology and Approach
To achieve these objectives, we use a visual question answer dataset, where annotation was made thorugh non-visual msnesoric real robotic tasks, fulfilling tasks. It contains images, as well as annotations for multi and single view task state detection. Additionally we anaylze in which setting VLMs perform better such tasks.
      

5. Timeline and Milestoes
First, we will select, clean, further annotate the dataset and build a pipeline to analyze our VLMs. Then we 
To fulfill our requirements, we select fitting questions and manually select 




## Seperation of tasks
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


## Paper, what to know?



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



### Points to mention
2 scenarios
failure modes 
detect current state of described task [possible answers]
multi view scenery
input multiple visual inputs 
select, clean and further annotate data
frontier models (multiple?)




# Porject proposal:

Recent advancements in Vision-Language Models (VLMs) have demonstrated significant potential for high-level reasoning. This project investigates the performance of frontier VLMs in the specific context of robotics, focusing on failure mode identification and multiview consistency. By utilizing a benchmark dataset of real-world robotic tasks, we will evaluate the capabilities of models like Qwen, Pi, and RT-2 in spatial and state reasoning. Our goal is to pinpoint current architectural weaknesses and provide actionable insights for improving VLM reliability in autonomous manipulation.
2. Problem Statement

Frontier VLMs are increasingly used to derive task instructions from visual inputs. However, their application in robotics faces a critical bottleneck: the "grounding gap." To be effective, a model must correctly interpret the current state of an environment, particularly when a task deviates from the expected path. Current models often fail to:

    Identify failure triggers: Recognizing why a grasp failed or an object slipped.

    Maintain temporal-state awareness: Understanding the transition between different task stages.

3. Project Goal and Objectives

This research focuses on two primary investigative tracks to determine if VLMs can provide a robust feedback loop for robotics:

    Track 1: Robotic Failure Modes: We test if VLMs can analyze a scene to detect the current operational mode and identify specific failure types (e.g., collisions, misses, or slips).

    Track 2: Multi-view Consistency: We examine if VLMs can extract more accurate spatial information when provided with multiple camera perspectives. We will measure if "multi-perspective" input significantly increases the model's effectiveness in detecting object coordinates and state changes.

The evaluation will require models to identify and select the correct state from a set of in-context options, using visual annotations generated through scene-image editing.
4. Methodology and Approach

The project utilizes a specialized Visual Question Answering (VQA) dataset. Unlike standard datasets, our annotations are derived from non-visual robotic sensors (such as force-torque and joint encoders), providing an objective ground truth for task success or failure.

    Data Preparation: Images are paired with multi-view and single-view annotations to test spatial reasoning.

    Model Selection: We will benchmark Qwen against robotics-specific models like RT-2 to compare general-purpose reasoning with domain-specific fine-tuning.

    Performance Metrics: We will analyze the delta in accuracy between single-view and multi-view prompts to quantify the benefit of perspective diversity.

5. Timeline and Milestones

    Dataset Refinement: Select, clean, and further annotate the sensor-grounded dataset.

    Pipeline Construction: Build the software interface to feed multi-view images and prompts to the target VLMs.

    Benchmarking & Selection: Manually select fitting edge-case questions to challenge model logic.

    Analysis & Reporting: Synthesize the data to identify which settings (single vs. multi-view) yield the most reliable robotic state detection.