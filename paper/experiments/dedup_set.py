# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""Hand-written test set for memory deduplication (experiment E13).

Each row is (fact, paraphrase_1, paraphrase_2, near_miss). Paraphrases state
the same memory in other words and should be merged into the fact. The near
miss is about the same topic with a different value and must stay separate.
"""

ROWS: list[tuple[str, str, str, str]] = [
    ("The user prefers short answers.", "Keep replies brief for this user.", "This user likes concise responses.", "The user prefers detailed, long answers."),
    ("The project deadline is 14 March.", "The project is due on March 14.", "We have to deliver the project by 14 March.", "The project deadline is 2 April."),
    ("The user's name is Priya.", "The person I am talking to is called Priya.", "User goes by the name Priya.", "The user's manager is called Priya."),
    ("The team uses PostgreSQL for the main database.", "Our primary database is PostgreSQL.", "The main data store for the team is Postgres.", "The team uses MongoDB for the main database."),
    ("The user works in the Asia/Kolkata time zone.", "The user's time zone is IST, Asia/Kolkata.", "Schedule things for the user in Indian Standard Time.", "The user works in the Europe/London time zone."),
    ("Deployments happen every Friday afternoon.", "We deploy on Friday afternoons.", "Friday afternoon is the regular release slot.", "Deployments happen every Monday morning."),
    ("The user is allergic to peanuts.", "The user has a peanut allergy.", "Peanuts cause an allergic reaction for the user.", "The user is allergic to shellfish."),
    ("The API rate limit is 100 requests per minute.", "Clients may send at most 100 requests each minute.", "The rate limit on the API is one hundred calls per minute.", "The API rate limit is 1,000 requests per minute."),
    ("The user writes code in Python and TypeScript.", "Python and TypeScript are the user's programming languages.", "The user codes mainly in TypeScript and Python.", "The user writes code in Rust and Go."),
    ("The quarterly budget is 2 million rupees.", "This quarter's budget is two million rupees.", "We have Rs 20 lakh to spend this quarter.", "The quarterly budget is 5 million rupees."),
    ("The user wants answers in British English.", "Use UK spelling when writing for the user.", "The user prefers British spelling and phrasing.", "The user wants answers in American English."),
    ("The staging server runs on port 8080.", "Port 8080 is used by the staging server.", "Staging listens on port 8080.", "The production server runs on port 443."),
    ("The user's daughter is called Meera.", "Meera is the user's daughter.", "The user has a daughter named Meera.", "The user's sister is called Meera."),
    ("Weekly team meetings are on Tuesday at 10 am.", "The team meets every Tuesday at 10 in the morning.", "Our weekly sync is Tuesdays, 10 am.", "Weekly team meetings are on Thursday at 4 pm."),
    ("The user dislikes emojis in replies.", "Do not use emojis when answering this user.", "The user does not want emoji in responses.", "The user enjoys emojis in replies."),
    ("The client is based in Singapore.", "Our client's office is in Singapore.", "The customer operates out of Singapore.", "The client is based in Dubai."),
    ("Invoices are due within 30 days.", "Payment terms are thirty days from the invoice.", "Clients must pay invoices in 30 days.", "Invoices are due within 7 days."),
    ("The user runs Ubuntu 24.04 on a laptop.", "The user's laptop has Ubuntu 24.04 installed.", "Ubuntu 24.04 is the operating system on the user's laptop.", "The user runs Windows 11 on a laptop."),
    ("The survey area covers 12 square kilometres.", "The area to be surveyed is 12 km².", "Our survey spans twelve square kilometres.", "The survey area covers 45 square kilometres."),
    ("The user is vegetarian.", "The user does not eat meat.", "The user follows a vegetarian diet.", "The user is vegan and avoids dairy."),
    ("Backups run every night at 2 am.", "A nightly backup starts at 2 in the morning.", "We back up every night at 02:00.", "Backups run every Sunday at midnight."),
    ("The user prefers dark mode in every app.", "Always suggest the dark theme to the user.", "The user likes dark mode interfaces.", "The user prefers light mode in every app."),
    ("The drone flight altitude is 120 metres.", "Drones fly at 120 m for the survey.", "The survey flight height is one hundred and twenty metres.", "The drone flight altitude is 60 metres."),
    ("The user is learning Japanese.", "The user is studying the Japanese language.", "Japanese is the language the user is currently learning.", "The user is learning German."),
    ("The repository's default branch is main.", "Main is the default branch of the repo.", "The repo uses main as its default branch.", "The repository's default branch is develop."),
    ("The user's favourite editor is VS Code.", "The user likes to work in Visual Studio Code.", "VS Code is the editor the user prefers.", "The user's favourite editor is Vim."),
    ("The conference talk is 20 minutes long.", "The user has a 20-minute slot for the talk.", "The talk at the conference lasts twenty minutes.", "The conference talk is 45 minutes long."),
    ("The office is closed on public holidays.", "No one works at the office on public holidays.", "Public holidays are office closure days.", "The office is open on public holidays."),
    ("The user takes notes in Markdown.", "Markdown is the user's note-taking format.", "The user writes all notes as Markdown files.", "The user takes notes in Microsoft Word."),
    ("The map projection for the project is UTM zone 44N.", "Use UTM zone 44N as the projection.", "The project works in UTM 44 north.", "The map projection for the project is UTM zone 43N."),
    ("The user's son plays cricket on Saturdays.", "Every Saturday the user's son has cricket.", "The user's son goes to cricket practice on Saturdays.", "The user's son plays football on Saturdays."),
    ("Error logs are kept for 90 days.", "We retain error logs for ninety days.", "Logs of errors are stored for 90 days.", "Error logs are kept for 7 days."),
    ("The user prefers meetings before noon.", "Schedule the user's meetings in the morning.", "The user likes to have meetings before 12 pm.", "The user prefers meetings after 3 pm."),
    ("The model is trained on Sentinel-2 imagery.", "Sentinel-2 images are the training data for the model.", "We train the model with Sentinel-2 satellite imagery.", "The model is trained on Landsat 8 imagery."),
    ("The user is a vegetarian who eats eggs.", "The user eats eggs but no meat.", "The user is an egg-eating vegetarian.", "The user is a vegetarian who avoids eggs."),
    ("The user's car is a white Honda City.", "The user drives a white Honda City.", "A white Honda City is the user's car.", "The user's car is a red Hyundai Creta."),
    ("Support tickets must be answered within 4 hours.", "Reply to support tickets in under four hours.", "The response time target for tickets is 4 hours.", "Support tickets must be answered within 24 hours."),
    ("The user wants weekly summaries on Monday.", "Send the user a summary every Monday.", "The user expects a weekly recap on Mondays.", "The user wants monthly summaries on the first day."),
    ("The GPU on the workstation is an RTX 4090.", "The workstation has an RTX 4090 graphics card.", "An RTX 4090 is installed in the workstation.", "The GPU on the workstation is an RTX 3060."),
    ("The user lives in Tiruchirappalli.", "The user's home city is Tiruchirappalli.", "The user is based in Trichy, Tamil Nadu.", "The user lives in Bengaluru."),
]
