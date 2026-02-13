# Bot or Not Challenge

## Overview

People make big and small decisions based on what they consume on social media. Who they vote for, what they buy, how they talk, and what they value - these are all shaped by what appears on our social media feeds. As a result, there is power in controlling what people see on social media.

Influence bots - programs that pose as humans on social media and push a particular agenda - are an increasingly common way of shaping the information people see and changing people's behavior.

We're headed towards a world in which we have no hope of knowing which posts are made by humans and which are made by bots, acting on behalf of organizations with questionable agendas. This is not a future in which we can thrive.

In this competition, you're invited to move us towards a different future - a world in which we know which social media accounts are run by bots and can label, regulate, and handle them as best serves the interests of society.

Your team will build a bot detection system that identifies accounts that are run by bots. But, be careful! Finding bots is good, but wrongly identifying human accounts as bots does real damage - isolating, othering, and disenfranchising people.

## Competition Details

### Objective

Build a bot detection system that consumes a time-ordered corpus of tweets and correctly identifies the accounts that are bots without wrongly flagging many human accounts.

### Participation

The competition is open to anyone! Any age, any education level, any skill level. You can participate individually or in teams of up to 4.

### Registration

To register fill the following google form: https://forms.gle/Eu56VUhb122vSYNLA

There is no hard registration cutoff. You can register up to the submission deadline.

### Key Dates & Timeline

- February 6th, 2026: Bot or Not Challenge Begins
- February 6th-14th, 2026: Detector Build Window
- February 14th, 2026, at 12:00 PM EST: Final Evaluation Datasets Released
- February 14th, 2026, at 1:00 PM EST: Submission Deadline
- February 15th, 2026: Results & Challenge Winners Announced

### Competitor Resources

Each team will be provided with four annotated social media datasets (two in English, two in French) which consist of Twitter posts made by several hundred accounts. Most of these accounts are human, but some belong to bots. The bot accounts are identified, allowing teams to test their detector's performance as they build it.

### The Task

Develop a bot detection program that consumes a dataset and identifies bot accounts. The program can be written however you see fit: a jupyter notebook, a python script, a Javascript microservice, or anything else.

For the purpose of this competition, we won't be reviewing your code - only the accounts your system identifies.

### Scoring

A bot detectors score on a given dataset is computed as follows:

- +4 for each bot account correctly detected (True Positive)
- -1 for each bot account not detected (False Negative)
- -2 for each non-bot account flagged as a bot (False Positive)

The rationale behind this scoring is as follows:

- It's important to find bots
- It's equally important to avoid flagging humans as bots
- Wrongly flagging a human is worse than missing a bot

### Final Evaluation

On February 14th, at 12:00 PM EST, we will send each participating team two final evaluation datasets - one in English and one in French. Each team will have until 1:00 PM EST to submit a list of the accounts that they believe are bots (for whichever language they are competing in).

A score will be prepared for each team's submission for each language dataset.

### Final Submission

Each team will have to submit their final results by email to the email address bot.or.not.competition.adm@gmail.com before 1:00 PM EST on February 14th.

Each submission needs to have the link to your GitHub code repo and the text file(s) containing the list of the accounts that you believe are bots.

**For the repo:**

- Our team needs to be able to access your GitHub repo link to see your code. We will only use it to ensure the integrity of your submission and for our own analysis and understanding, we won't share it.
- In your GitHub repository you need to have a README file that explains how your code works.

**For the text file:**

- It has to be called [team_name].detections.[lang].txt where you replace [team_name] by your team name and [lang] by en if it's your english dataset submission and fr if it's your french dataset submission.
- In this file you will have a list of all the user IDs of the accounts that you believe are bots. There is one user ID per line.
- To see an example of the file format it should be the same then the dataset.bots.txt file we gave you.
- You can submit [team_name].detections.en.txt and/or [team_name].detections.fr.txt, you don't have to submit both.

In the event where you made a mistake and you want to quickly correct it and resubmit before the deadline we will only keep the last submission of each team.

### Prizes

- Best in French: 500$
- Best in English: 500$
- Best in Both: 1000$

## Helpful Guidelines

### Q&A

**Do I have to build software? Can we just be very good human bot detectors?**

- Technically, yes. However, in the final evaluation, you'll only have 1 hour from the time you receive the evaluation dataset to submitting your scores... which isn't a lot of time to do more than run a script. If you end up using any human interaction with your code please mention it in the README.

**Are the bots in the testing datasets representative of the bots in the final evaluation data?**

- Yes. The vast majority of the bot algorithms used to produce the final evaluation data are the same as those used to produce the testing datasets. However, we will add a few additional bot algorithms for the final evaluation dataset that your systems haven't seen before. While different, these will be of the same level of sophistication as the ones used to produce the testing data.

**Does my team have to build a detector that works on English AND French social media data?**

- No. Your team can focus on one or both. From our past experience, we've found that effective bot detectors are quite different between languages. So if you do focus on both, then you'll likely need to build two different detectors... which could be quite ambitious for the amount of time you'll have.

**I've registered late, how can I get access to the practice datasets?**

- We will verify once a day if there are new registrations and send them the datasets by email. You can also send us an email at bot.or.not.competition.adm@gmail.com if you still haven't received it.

**Who can I contact if I need help?**

- If you have any questions related to the challenge you can send an email to bot.or.not.competition.adm@gmail.com and we will respond to you as soon as we can!

## Technical Details

### Social Media Dataset

Each social media dataset is a file with one tweet per line. The tweet is JSON-formatted. It contains the following fields:

- **text**: The text content of the tweet.
- **created_at**: The specific date and time at which the tweet was posted.
- **id**: The unique ID of the tweet in the dataset.
- **author_id**: The author ID of the author of the tweet. Each author has a unique ID.
- **lang**: The language associated with the tweet.

### How the datasets were collected

We had a set of topics that we chose and for each of them a set of specific words or hashtags we were looking for in tweets over two days. Once we had these first tweets gathered we made a set of all the users associated with these tweets. Then for every user we gathered up to 100 of their tweets during the two days we had selected. Finally we cleaned up the dataset of any obvious bots, anonymized the mentions and the url associated with tweets.

### What the participants are given

**dataset.tweets.json**

This is the set of all the tweets from the original dataset with bots tweets inserted into it. The tweets are as described above. Each user publishes between 10 and 100 tweets.

**dataset.users.json**

This is the set of all users that are present in the dataset.tweets.json set. You get the following data on each user:

- **id**: The unique user ID. (This ID is the same then the author_id)
- **username**: Username used on Twitter and that appears beside their tweets. It has to be unique.
- **name**: The name of the user.
- **description**: The short description of the user written by themselves. This field can be left empty and can contain anything.
- **location**: This is the location that the user assigns to themselves, it can be anything (even Fairyland) and does not need to be filled.
- **tweet_count**: The total amount of tweets of the user in the dataset.
- **z_score**: The z-score associated with the user according to its tweet_count.

**dataset.bots.txt**

This text file contains the list of the user ID of all the known bots present in the dataset. There is one user ID per line. This file will only be available with the practice datasets.

### Bot Detector Output File

The bot detector should output a file with one account number per line. Each account number should correspond to one that the detector has flagged as a bot. (This format is the same then the dataset.bots.txt file)

---

Good luck everyone! May the best detectors win!
