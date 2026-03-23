
<div align="center">

# ☠️ ONE PIECE MEDIAPIPE

### _Real-Time Pose Detection meets the Grand Line_

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-000000?style=for-the-badge&logo=python&logoColor=green)

> _Inspired by my favorite anime series, One Piece_

</div>

---

## 🌊 What Is This?

MediaPipe tracks your every move in real-time — and the moment you strike a pose worthy of the **Straw Hat Crew**, the program fires back with iconic voice lines straight from One Piece.



---

## 🏴‍☠️ Characters & Triggers

| Character | Pose Trigger             | Reaction |
|-----------|--------------------------|----------|
| 🔴 **Monkey D. Luffy** | Squad with one punch down| **Gear Second** steam aura + red screen flash |
| 🔵 **Franky** | Arms raised overhead     | **"SUPEEEER!"** + blue energy pulse |
| 🌸 **Nico Robin** | Arms crossed over chest  | **Fleur** petal effects + whisper audio |
| ❓ **???** | Coming soon...           |  |

---

## ⚙️ How It Works

```
[Webcam] → [OpenCV Frame] → [MediaPipe 33 Landmarks] → [Pose Matcher] → [REACTION!]
```

1. **Webcam Input** — OpenCV grabs your live camera feed
2. **Pose Detection** — MediaPipe maps 33 body landmarks in real-time
3. **Pattern Matching** — Landmark positions are compared against pose signatures
4. **Reaction Fires** — Voice line plays via Pygame, visual effects overlay your body

---

## 🔧 Tech Stack

- 🐍 **Python 3**
- 📷 **OpenCV** — frame capture & visual rendering
- 🦴 **MediaPipe** — real-time body landmark detection
- 🔢 **NumPy** — landmark math & pose logic
- 🎵 **Pygame** — audio playback for voice reactions

---

## 📁 Project Structure

```
onepiece-mediapipe/
├── onepiece_pose/
│   ├── main.py                    # Entry point — run this
│   ├── test_mp.py                 # Pose testing & debugging
│   ├── pose_landmarker_lite.task  # MediaPipe model
│   └── audio/
│       ├── gearsecond.wav         # 🔴 Luffy's Gear Second
│       ├── supeer.wav             # 🔵 Franky's catchphrase
│       └── fleur.wav              # 🌸 Robin's Fleur
└── .gitignore
```

---

## 🚀 Get Started

```bash
# Install dependencies
pip install opencv-python mediapipe numpy pygame

# Set sail
python onepiece_pose/main.py
```

> Make sure your webcam is connected and you have enough space to **strike a pose** ⚡

---

## 🗺️ Roadmap

- [ ] More Straw Hat crew members with unique pose triggers
- [ ] ML-powered pose classification (beyond rule-based matching)
- [ ] Advanced particle effects — actual smoke, fire, electricity
- [ ] Character selection UI — choose your crew member
- [ ] Multi-person detection — Straw Hat crew battles in real-time

---

## 🎌 Visual Effects

When a pose is recognized, the system can trigger:

- **Voice lines** — character-authentic audio samples
- **Aura effects** — glowing energy overlaid on your body silhouette
- **Smoke/steam visuals** — Gear Second style particle bursts
- **Screen flash** — color-coded reactions per character

---

<div align="center">

 by **Irem Erturk**


</div>
