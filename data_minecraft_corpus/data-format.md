<a name="_m447vebe5s4z"></a>Data Format for CwC Minecraft Demos

# <a name="_urb4h5bxc4mz"></a><a name="_8xzrt9jtlep3"></a>Directory Structure
**Note:** The terms “world state” and “observation” are used interchangeably and refer to the same thing.



Each directory is a collection of all game logs and screenshots collected on the date specified in the directory name.

In a directory, say “data-4-16” (after unzipping the two .zip files within):

- The game logs are in the “logs” subdirectory
- The screenshots are in the “screenshots” subdirectory

The directories within “logs” and those within “screenshots” have a 1:1 correspondence. Each such directory (named by a unique experiment/game ID) contains the data collected for a specific game played. An experiment ID, say something like “B36-A35-C17-1523916463263”, contains 4 parts to it -- the builder ID (B36), the architect ID (A35), the target structure (C17) and a timestamp (1523916463263).

Within such a directory on the logs side, say “data-4-16/logs/B36-A35-C17-1523916463263” you’ll find the game log. Within the corresponding one on the screenshots side, “data-4-16/screenshots/B36-A35-C17-1523916463263” you’ll find all the screenshots.
# <a name="_7orm334knrqg"></a>Screenshots data
Three different types of screenshots are taken for each observation:

- 1 from the architect’s perspective
- 1 from the builder’s perspective
- 1 each from 4 fixed view perspectives -- in each of the 4 canonical directions around the build region
# <a name="_5dmhbvr1paur"></a>Log data
The main file to access the log data collected is aligned\_observations.json. Its format is as follows:

**aligned-observations.json**

Basically, this is a chronological list of all observations. An observation gets recorded when:

- When the builder places a block
- When the builder picks up a block
- When a chat message is sent by either architect or builder

***The top level dictionary:***
- ***WorldStates***: List of all world states recorded in chronological order (more below)

- ***NumFixedViewers***: Number of fixed view perspectives used to collect additional screenshots (always 4 for us) apart from the architect and builder perspectives

- ***TimeElapsed***: Total time taken for the game from start to finish in seconds

***WorldStates:*** Each item in the list is the world state at a certain point in time. It is as follows:

- ***TimeStamp***: The exact time when this observation was recorded

- ***BuilderPosition***: Builder’s <x, y, z, yaw, pitch> coordinates

- ***BuilderInventory***: List containing 6 items corresponding to all 6 block colors available in the game. For each color, it stores the number of blocks of that color that the builder currently possesses.

- ***BlocksInGrid***: All blocks currently placed in the build region. Each block has a color and 2 sets of <x, y, z> coordinates -- absolute and perspective. The latter is relative to the builder’s perspective, i.e, the builder’s <x, y, z, yaw, pitch> coordinates (recorded in BuilderPosition above).

    The build region is an 11 x 9 x 11 grid, where:

    - -5 <= x <= 5
    - 1 <= y <= 9 (where y=1 is ground level)  (**Important! y is the 3D vertical axis in the Minecraft world!)**
    - -5 <= z <= 5

- ***ChatHistory***: The entire dialog history up until this point in time

- ***Screenshots***: All screenshots taken for this observation. There are 6 fields for the 6 screenshot perspectives -- builder, architect and the 4 fixed viewers. Each has a value which is the name of the screenshot image in the corresponding screenshots directory. In the case where there was no screenshot taken or was taken but couldn’t be aligned to this observation, the value for the corresponding field is null.

- ***DialogueStates*** (\*\*Builder demo only\*\*): If the dialogue manager was involved in directing the flow of information through the system at that point in time, the DialogueStates field will contain a list of dialogue states that the system transitioned through as well as their arguments (specific to the dialogue state; i.e., a resulting plan from a PLAN state, or a semantic parse from a PARSE state). The information consists of: the name of the Dialogue State (e.g. PARSE\_DESCRIPTION, PLAN, REQUEST\_VERIFICATION etc.); the input text (from the human Architect); the output text (from the system as Builder); the resulting semantic parse produced by the parser; the response from the planner during planning; and the execution status of the provided plan.


# <a name="_vznrf9pa761r"></a>Other data
In a directory, say “data-4-16”, you’ll also find some other files. They are as follows:

- **dialogue.txt** -- The dialogue history for all game logs collected on the date specified in the directory name -- in human-readable format
- **dialogue-with-actions.txt** -- The dialogue history **and** builder actions for all game logs collected on the date specified in the directory name -- in human-readable format
- Within the **pdf-simplified** subdirectory:
  - Each pdf file contains info on all games played by a certain pair of builder and architect on that day. For example, “B51-A54-simplified.pdf” contains info on all games played by builder #51 and architect #54 on April 16. It contains one chapter per game played. A chapter is titled by the target structure used in that game. It has two sections -- one to show screenshots of the target structure used and one to show the dialog history.
