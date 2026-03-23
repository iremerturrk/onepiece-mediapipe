import mediapipe as mp

print("mp =", mp)
print("mp file =", getattr(mp, "__file__", "NO_FILE"))
print("dir has solutions =", "solutions" in dir(mp))
print("dir(mp) =", dir(mp))