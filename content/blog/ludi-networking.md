---
author: "Vien Vuong"
title: "LUDI Computer Networking Community"
date: "2022-01-16"
description: "LUDI (Library for University Distance Instruction) aims to create a web service consisting of a user-friendly database of computer networking educational sources to provide an organized space for university educators to discover relevant software tools."
tags: ["webdev", "project"]
comments: false
socialShare: false
toc: false
cover:
  src: /ludi-networking/landing.png
  alt: LUDI Computer Networking Community
---

[**Check it out!**](https://ludi.cs.illinois.edu)!

## Overview

As the COVID-19 pandemic resulted in over a billion students departing classrooms, students and instructors adapted to the circumstances by relying heavily on online educational tools. While the pandemic is decreasing in severity, the use and utility of such programs is growing exponentially.

LUDI (Library for University Distance Instruction) aims to create a web service consisting of a user-friendly database of computer networking educational sources to provide an organized space for university educators to discover relevant software tools.

Current features include uploading, searching, reviewing, saving favorites, and commenting on resources. Users are also able to create their own accounts (or link their university accounts), and log in to start rating, saving resources, and join the discussion forum. Each user has their own profile page where they can see their favorite resources, and create lists of resource recommendations.

Future plans include exploring user behavior by collecting usage data with Google Analytics and online surveys. Variables to be investigated include popular searches, visits to product pages per search, visits to download links per search, and peak usage times for the time being. This data will then be used to curate resources for each user.

LUDI is founded by Professor Matthew Caesar in the Department of Computer Science at UIUC. LUDI is sponsored and supported by the Association for Computing Machinery's Special Interest Group on Data Communications, the Darin Butz Foundation, the University of Illinois Department of Computer Science, and the University of Illinois Department of Electrical and Computer Engineering.

LUDI was presented at SIGCOMM 2021 at UIUC.

LUDI is built with a ReactJS frontend, and a NodeJS Express backend that is hosted on a CentOS Linux server.

## How to use LUDI?

### Searching

To search for resources, either enter text into the search bar or select a tag to search by.

1. Selecting search fields (title, description, category, author) determines which fields of the resources we check for matches to the text in the search bar.

2. Selecting tags will search within the tags/categories field of the resources. If the tags search has ‘AND’ selected, only resources that have all the tags are returned. If ‘OR’ is selected, resources that have any of the tags can be returned.

### Signing up and logging in

1. To sign up, click the ‘Sign Up’ button in the top right corner of the webpage. After you enter the required fields and click ‘Create my account’ a new account will be created if the email isn't already taken. Reload the page to check if you have been logged in successfully.

2. To log in, click the ‘Sign In’ button and enter your information. To verify you’ve been logged in successfully, reload the page.

3. To sign out, click your name in the upper right corner of the page and click ‘Sign out’

### Saving and viewing saved resources

1. To save a resource, navigate to any resource and click on the bookmark icon. You must be signed in to save a resource.

2. To view your saved resources, click on your name or avatar in the upper right corner or the webpage and then click ‘Profile’ in the dropdown options.

### Submitting new resources

To submit a new resource, click ‘Submission’ in the footer of the webpage. Fill out the embedded form and then click ‘Submit’.

### Getting help

If you have comments, suggestions, or need help, please email us at ludi-help@illinois.edu

## Images

![Landing Page](/ludi-networking/landing-light.png)
![Categories Page](/ludi-networking/categories.png)
![Favorites Page](/ludi-networking/favorites.png)
![Profile Page](/ludi-networking/profile.png)
![Login Page](/ludi-networking/login.png)
![Registration Page](/ludi-networking/registration.png)
![Search Page 1](/ludi-networking/search1.png)
![Search Page 2](/ludi-networking/search2.png)
![Search Page 3](/ludi-networking/search3.png)
![Search Page 4](/ludi-networking/search4.png)
![Resource Page](/ludi-networking/resource.png)
