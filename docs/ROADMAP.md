# Roadmap

Ordered by expected value.

0. **Run the manual hardware checklist** ([VALIDATION.md §7](VALIDATION.md#7-manual-hardware-validation-checklist)) in both engine modes.
   - Done:
     - engine-level camera runs (15 FPS at 720p, with MJPG and without);
     - live-camera UI passes in both modes: Phase 4 headless, Phase 5 in a **visible** window with real gestures (OPEN_PALM, FIST, PINCH and POINT drove their commands).
   - Still open:
     - gesture accuracy against labelled intent (a person records what they performed);
     - swipes on a real camera;
     - unplugging the camera mid-session.
   - Follow [VALIDATION.md §10](VALIDATION.md#manual-checklist-requires-a-person-at-the-webcam).
   - Investigate the 15 FPS capture limit: lighting and auto-exposure, 640×480, `cameraApi: msmf`. Measure with `npm run bench:python`.
1. **Verify the agent against the live provider:** set `ANTHROPIC_API_KEY`, run `npm run agent:smoke`, then the manual agent checklists ([VALIDATION.md §8](VALIDATION.md#8-plan-phase-3-of-5-ai-agent) and §9). Only mocked-provider tests have run so far. Check whether the real model picks `app_diagnostics` and `run_workflow` when appropriate.
   - Profile the possible cost of the panel's backdrop blur in browser mode (one run: 52–55 vs 39–45 FPS; another: no difference).
2. **Move browser-engine inference to a Web Worker.** (The Python engine already keeps inference off the browser's main thread.)
   - **Measured cause (Phase 5):** a CDP CPU profile shows 61–82% of main-thread time in WebGL `useProgram` / `readPixels`. That is MediaPipe's GPU delegate waiting synchronously for the GPU. Actual WASM compute is about 3%, three.js about 1%, and app code under 1%.
   - **Ruled out:** the CPU delegate is slower here (51 ms vs 31–39 ms per inference, render 18 FPS).
   - **The fix:** transfer `ImageBitmap`s to a module worker running HandLandmarker (GPU via `OffscreenCanvas`), with a bounded queue of size 1 (latest frame wins). The wait then happens off the main thread, and render FPS stays at display rate.
   - **Reproduce:** `node scripts/integration-check.mjs --video hand.mjpeg --seconds 8 --cpu-profile`, then read `cpuProfile` in `report.json`.
3. **Embodied gestures, live follow-up** ([VALIDATION.md §12](VALIDATION.md#12-embodied-interaction-face-gaze-continuous-spin-cadence-walking-emotions)):
   - re-run `--guided embodied` with a person who follows the step prompts;
   - thumb up / down were **not recognised live** in either engine: record landmarks for a few real thumbs-up / down poses and recalibrate `fold4` / the direction ramp;
   - check the new `minReach` walk rule live (WALK was recognised during spin steps before it was added);
   - observe hand loss *while walking* and `L` *while spinning* on the real camera;
   - profile the cost of browser-engine face detection (about 5–10 ms every 120 ms) on slower GPUs.
4. **Live-camera validation session.** Run the hardware checklist in VALIDATION.md. Then record a small, consented, labelled clip set (landmarks plus labels only, no video) to measure per-gesture precision and recall and to retune the thresholds.
5. **Learned static classifier.** Train a small MLP on normalised landmarks from that dataset behind the existing `classify()` contract. Report its accuracy against the rule baseline on held-out data. *Done in §14 with public data (HaGRIDv2) instead of a project-recorded clip set; the live, consented evaluation is still open.*
6. **One-Euro filtering of landmarks.** Smooth the jitter that the overlay and robot look-at visibly inherit from the model, and the swipe velocity estimate.
7. **Real robot asset.** Use a licensed GLB with rigged `idle`, `wave` and `spin` clips, and map head and arm bones for look-at and aim.
8. **Per-user calibration.** A short guided capture of open palm and fist that adapts the extension thresholds.

## After the deep-learning upgrade (VALIDATION.md §13)

1. **Live accuracy, properly labelled:** re-run `--guided perception` with someone who follows each prompt, and ideally with a second person labelling. Only then report gesture / expression precision, recall and false-activation rate.
2. **Expressions:** confirm live that `mouthSmile` and the other blendshapes respond. The smile step stayed at 0.00 in the session; check whether that was non-adherence or a calibration issue, then tune `interaction.expression`.
3. **Reference poses:** the video's puppeteer poses are mostly `None` for the canned gesture model. If the open-palm / palm-down / pinched-index states must be *classified*, train a small custom gesture classifier with MediaPipe Model Maker on consented, labelled landmark data. *Partly done in §14: a hand-pose network trained on HaGRIDv2 (THREE_FINGERS and an OTHER class).*
4. **Legged mecha:** the reference robot has legs. A licensed rigged GLB (walk, kick, lift clips) would replace the hovering procedural robot; the flag must then be re-attached to that model's chest bone.

## After the hand-pose network (VALIDATION.md §14)

1. **Live check of walk / run / squat / spin and face mirroring** with the physical webcam (visible window, explicit permission). None of these is verified live yet.
2. **Subject-independent evaluation:** the mirror has no subject ids, so the 80 / 20 split may put the same person in both halves. Download the official HaGRIDv2 annotations (user ids) when the official host is reachable and re-split by subject.
3. **More data for the small classes:** THREE_FINGERS, THUMB_UP / DOWN and the negatives have only 200 images each (partial shards on a slow network). `fetch_dataset.py` without `--row-groups` downloads the full shards.
4. **Remaining false activations:** the OK sign is a real thumb-index touch and reads as PINCH; "mute" (index on the lips) is called Pointing_Up by the canned model. Options: let a confident net OTHER veto the canned POINT, and require the pinch tap to start from an open hand.
