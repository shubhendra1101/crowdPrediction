# Correcting head labels in CVAT (T1.5 → T1.6)

You get 100 frames from your classroom camera with a dot already placed on most heads by a model.
Your job: **make sure there is exactly one dot on every visible head — no more, no less.**
Expect roughly 3–5 hours in total. You can stop and continue later; CVAT saves your work.

Files (on the laptop, never uploaded to GitHub):
- `data/cctv/labels/images.zip` — the 100 frames
- `data/cctv/labels/cvat_prelabels.zip` — the pre-placed dots

## 1. Create the task

1. Go to **app.cvat.ai** and log in.
2. **Projects → +** is not needed; go to **Tasks → + → Create a new task**.
3. Name: `crowdsafe_classroom_heads`.
4. Under **Labels**, click **Add label**: name **`head`**, type **Points**. Click **Continue**.
5. Under **Select files**, upload **`images.zip`**. Click **Submit & Open**.

## 2. Load the pre-placed dots

1. In the task page, click **Actions → Upload annotations**.
2. Format: **CVAT 1.1**. File: **`cvat_prelabels.zip`**. Click **OK**.
3. Open the job (click **Job #…**). You should see dots on the heads.

## 3. Correct each frame

| Situation | What to do |
| --- | --- |
| Head with no dot | Press **N** (or pick the Points tool, label `head`), click the **centre of the head** once, press **N** again to finish |
| Dot not on a head (bag, chair, poster) | Click the dot, press **Delete** |
| Two dots on one head | Delete one |
| Dot on the body, not the head | Drag it onto the head centre |
| Head partly hidden but you can tell a person is there | Put a dot where the head is |
| Person fully hidden | No dot |
| Reflection in the window / person outside the room | No dot |

- Move between frames with **D** (next) and **A** (previous). Zoom with the mouse wheel.
- Work steadily; accuracy matters more than speed. In crowded frames, zoom in and go left to right.

## 4. Save and export

1. Press **Ctrl+S** often.
2. When all 100 frames are done: back on the task page, **Actions → Export task dataset**.
3. Format: **CVAT for images 1.1**. Untick "Save images". Click **OK** and download the zip.

## 5. Give it to the agent

Put the exported zip in `C:\Users\hp\OneDrive\Desktop\CrownPrediction\data\cctv\labels\` and say **"labels done"**.
The agent imports it, compares each model's count with your corrected count, and uses the frames for the fusion (T3.7).

**Privacy:** these frames show your classmates. Keep the CVAT task private, don't share the link, and delete the
task from CVAT when the project is finished.
