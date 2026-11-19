# Project Questions (Desktop Assistant)

Answering these will help define what to build and what to leave out.

## Product & scope
1. What is the core job of the assistant? (e.g. quick Q&A, running tasks on my Mac, controlling apps, coding help, something else?)
-> evrything, being modular for plugins
2. What are the first 3–5 things you want to be able to say/do with it on day one?
-> eg: open chrome and choose a circa 10 min video from my for you page
3. What should it explicitly *not* do (for safety/privacy/scope)?
-> it shouldnt do anything without permission, user can choose between security levels

## The “island” / UI
4. On notch Macs: should it sit *around* the camera notch like a Dynamic Island pill, or live in the menu bar? On non-notch Macs: menu bar icon, a pill near the camera area, or a floating window?
-> around the notch ore as a pill next to the cammera or in the native apple dynamic iland
5. How should it look/behave when idle vs. listening vs. thinking vs. showing a result? (collapsed pill, expanded panel, text transcript, etc.)
-> on the notch macs it should move metaball-like out of the notch like its living behind it, idle invisible (living behind the notch); on pill-macs it should move from the side of the cammera below it, idle as a pill in next to the cammera or just in the middle to of the screen if theres no cammera;; when activated like said before it shold pulsade with the spocken audio, while thinking it should become a loding circle, when doing something the "blob" should transform into a second cursor and while speaking it should pulsate to the spoken audio
6. Activation: you mentioned wake word and clicking the island/near camera — which is primary? What wake word/phrase do you want? Should it listen continuously or only after a click/hotkey?
-> the wakeword should be primary but you should be able to change the word or set it so it activates to two clics
7. Any other inputs/outputs? (hotkey, drag-drop, clipboard, notifications, sound cues, spoken replies)
-> hotkey: 3x shift fast, file drag and drop into the blob, clipboard read if the permission level alows it, sfx matching to evrithing it does, notifcation just if verbosity level is all the way down and you dont activly see what the assistand is doing so it notifies you if he needs a permit or has finished

## Voice, models & routing
8. Speech-to-text: fully on-device only, or is cloud STT acceptable? Any language(s) besides English?
-> multilingual, standart english, free cloud prefered for diffrent voices, if not possible fallback for on device stt
9. The “small self-trained on-device model” that decides chat vs. agent: do you already have a model/dataset/idea for it, or should that be designed? What kinds of requests count as “just answer with text” vs. “send to agentic harness”?
-> it should be just a small keras model with an maximum input of 5000 letters that get passed in by ther unicode, output should be one one bool agent or no; the dataset isnt there it has to be created by you.
10. For text answers (chat path): local LLM, a remote API (which?), or both? Any model preferences or size/latency constraints?
-> user should be able to connect there own provider. the app should on firs startup scan the system for apis that are already installt that can be reused.
11. For the agent path: which harness/CLI should it dispatch to (opencode, claude, codex, aider, other)? Should it run tasks automatically or ask for confirmation before executing anything?
-> https://github.com/badlogic/pi-mono

## Capabilities & permissions
12. What should the agent be allowed to do on your Mac? (run shell commands, read/write files, control apps/AppleScript, browse web, calendar/mail, etc.)
-> with the must-have permision for the task it should be allowd to do anything
13. Any data it must never access or send off-device?
-> data that the user flags as private

## Technical preferences
14. Python/C split: what do you want written in C vs. Python? (e.g. audio buffer/performance-critical bits in C, everything else Python — or something specific?)
-> performance critical things in c, importable into python, ui with the custob blob in c if not possible with python
15. Packaging/how you want to run it: `uv run` from source, a standalone `.app` bundle, a menu-bar/login item, or all of the above?
-> mainly on github as a .bin realese; later homebrew and .dmg realease auf github
16. Dependency/footprint constraints? (e.g. must stay lightweight, OK to pull large ML libs, prefer Apple MLX, avoid certain frameworks)
-> stay lightweight but if something needs the space use it
17. Python version preference and any libraries/tools you definitely want (or want to avoid)?
-> use newist if possibe but have in mind if something doesent work on lets say 3.14 we have to downgrade to 3.13

## Project setup (uv/git)
18. Do you want a git repo initialized now (and an initial commit of the structure), or later?
-> full git and uv setup
19. Should I create a `uv` project/venv now, or wait until dependencies are decided?
-> start a venv
20. Testing/quality from the start: pytest/ruff/mypy/CI, or keep it minimal for now?
-> pytest 100% coverage
21. License and project name (folder is currently “Desktop assistend”)?
-> standart licenc for stuff so its usable but it canot be copied

## Anything else
22. Are there apps/projects you like as reference (behavior or look), or things you definitely don’t want it to feel like?
-> dant make ai slop out of this use all your capabillities to get it right
23. What’s the biggest risk or annoyance you want to avoid with this project?
-> not having it finished
