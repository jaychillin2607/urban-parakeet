# Task: Statues
- Est. time/effort: 20 hours

## GitLab Repository
Please use the private repository to version control your code:
https://git.toptal.com/screening-ops/Nidhi-james-gopalkrishnan

- Username: Nidhi-james-gopalkrishnan
- Password: d1245452c93bf6e9a51b747c7cd9b399

## Task Description
Expected delivery time: 7 days

## Task scope and expectations
The purpose of this task is to analyze the data and build a model that solves the specified problem. We want to see how you:

- explore the data,
- identify the most important features,
- track experiments,
- summarize your findings,
- use the data to build a working AI solution.
  
We seek to assess your proficiency in the following:

- setting up a proper project infrastructure,
- demonstrating knowledge of simple problem-solving, and
- effectively utilizing your preferred frameworks and libraries.
  
The task has certain objective target metrics, but your approach and presentation are also taken into account.

Please make sure that your model is easy to run and reproduce.

## Task details
The following dataset contains images of statues, separated into three classes: statues of Vladimir Lenin, statues of Mustafa Kemal Ataturk, and all others.

https://drive.google.com/file/d/1vdXI9GUteObuV3Su7LOZqWpwaZ_YFPik/view?usp=sharing

- Your goal is to create a model that detects the head of the statue on the image and answers whether it is a statue of Lenin, Ataturk, or not.
- Your solution should contain a script that runs the pre-trained model on a set of images in the specified folder and creates a 'results.csv' file in the following format: `{image name};{x1};{y1};{x2};{y2};{class}`.
  - `x1, y1` are the coordinates of the top left corner of the head's bounding box (assuming `(0,0)` is the pixel in the top left corner of the image).
  - `x2, y2` are the coordinates of the bottom right corner of the head's bounding box.
  - `'class'` is `1` for Lenin statues, `2` for Ataturk statues, and `0` for others.
- If no head is found on the image, write `{image name};0;0;1;1;0`.
- If there are multiple heads, print only the coordinates of Lenin's or Ataturk’s head; if there are none or more than one, print the coordinates of the largest bounding box.

## Milestones and task delivery
- The deadline to submit your completed project is 1 week from the moment you receive the project requirements.
- It means the project code must be submitted within 1 week from the moment it was delivered to you by email.
- If you schedule your final interview after the 1-week deadline, make sure to submit your completed project and all code to the private repository before the deadline
- Everything that is submitted after the deadline will not be taken into consideration.
- To ensure sufficient review time, please commit all code or documents at least 6 hours before the meeting. Submissions after this time will not be considered.
- Please join the meeting room for this final interview on time. If you miss your interview without providing any prior notice, your application may be closed.

Submission Requirements
- This project will be used to evaluate your skills and should be fully functional without any obvious missing pieces.
- You have 7 days from the date you receive the brief to submit your completed project.
- If you schedule your nal interview after the 7-day deadline, make sure to submit your completed project and all code to the private
- repository before the deadline. Everything that is submitted after the deadline will not be taken into consideration.
- If your interview is scheduled before the 7-day deadline, your project outputs and all code must be submitted to the private repository at least 2 hours before the interview. Failure to meet this requirement will result in your interview being canceled, and
your application may be closed.
- After scheduling your interview, review the meeting details in the con rmation email carefully to be sure you are fully prepared for
the call.
- Choose a quiet, distraction-free environment for the call with a reliable internet connection and functioning audio/video equipment.
- You will be asked to share your screen, so we recommend joining from a computer.
- Test your technology (camera, microphone, and screen-sharing tools) beforehand to prevent technical issues during the interview.
- Arrive on time and be ready to engage—punctuality and preparation demonstrate professionalism and respect for everyone’s time.
- Please note that your application may be declined if you reschedule your interview less than 12 hours in advance, you’ve
rescheduled more than once, or you fail to attend your interview without prior warning.