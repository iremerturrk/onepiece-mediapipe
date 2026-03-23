"""
One Piece x MediaPipe – Pose Detector  v6
mediapipe >= 0.10.21 (Tasks API)

Model: pose_landmarker_lite.task  (main.py ile aynı klasörde)
"""

import cv2, time, math, random, os, sys
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ═══════════════════════════════════════════════════════
# MODEL & SES
# ═══════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
if not os.path.exists(MODEL_PATH):
    sys.exit("[HATA] pose_landmarker_lite.task bulunamadı!")

pygame.mixer.init()
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

def load_sound(fn):
    p = os.path.join(AUDIO_DIR, fn)
    try:   return pygame.mixer.Sound(p) if os.path.exists(p) else None
    except: return None

SOUNDS = {k: load_sound(v) for k,v in {
    "luffy_gear":   "gearsecond.wav",
    "franky_super": "supeer.wav",
    "robin_fleur":  "fleur.wav",
}.items()}

def play_sound(key):
    s = SOUNDS.get(key);
    if s: s.play()

# ═══════════════════════════════════════════════════════
# LANDMARK
# ═══════════════════════════════════════════════════════
LM = {
    "NOSE":0,"LEFT_SHOULDER":11,"RIGHT_SHOULDER":12,
    "LEFT_ELBOW":13,"RIGHT_ELBOW":14,
    "LEFT_WRIST":15,"RIGHT_WRIST":16,
    "LEFT_HIP":23,"RIGHT_HIP":24,
    "LEFT_KNEE":25,"RIGHT_KNEE":26,
    "LEFT_ANKLE":27,"RIGHT_ANKLE":28,
}

def extract_lm(raw, w, h):
    return {n:(int(raw[i].x*w),int(raw[i].y*h))
            for n,i in LM.items() if i<len(raw)}

def dist(a,b):      return math.dist(a,b)
def mid(a,b):       return ((a[0]+b[0])//2,(a[1]+b[1])//2)
def clamp(v,lo,hi): return max(lo,min(v,hi))

def angle(a,b,c):
    a,b,c=np.array(a,float),np.array(b,float),np.array(c,float)
    ba,bc=a-b,c-b
    n1,n2=np.linalg.norm(ba),np.linalg.norm(bc)
    if n1==0 or n2==0: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(n1*n2),-1,1))))

# ═══════════════════════════════════════════════════════
# COMPOSITING HELPERS
# ═══════════════════════════════════════════════════════
def alpha_blend(frame, layer, alpha_mask):
    """
    frame    : uint8 BGR  (H,W,3)
    layer    : uint8 BGR  (H,W,3)  – efekt rengi
    alpha_mask: float32  (H,W)    0..1
    """
    a3 = alpha_mask[:,:,np.newaxis]
    return np.clip(frame.astype(np.float32)*(1-a3) +
                   layer.astype(np.float32)*a3, 0, 255).astype(np.uint8)

def radial_gradient(cx, cy, radius, h, w):
    """Merkeze doğru yoğunlaşan 0..1 gradyan maskesi."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    return np.clip(1.0 - d/max(radius,1), 0, 1)

def gaussian_blur_fast(img, ksize):
    k = ksize if ksize%2==1 else ksize+1
    return cv2.GaussianBlur(img,(k,k),0)

# ═══════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████
#  LUFFY — GERÇEK STEAM / DUMAN
#  • Vücuttan yükselen beyaz/gri buhar sütunları
#  • Her partikül render = tam frame boyutunda yumuşak
#    Gaussian blob → gerçek volumetrik his
# ██████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════
class SteamPuff:
    def __init__(self, x, y):
        self.x     = float(x + random.gauss(0, 12))
        self.y     = float(y + random.gauss(0, 6))
        self.r     = random.randint(22, 55)
        self.max_r = self.r + random.randint(40, 90)
        self.life  = random.randint(35, 65)
        self.ml    = self.life
        self.vx    = random.gauss(0, 0.3)
        self.vy    = random.uniform(-1.2, -0.3)
        v          = random.randint(210, 250)
        self.color = np.array([v,v,v], dtype=np.float32)   # gri-beyaz

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy *= 0.97
        self.r  += (self.max_r - self.r) * 0.06   # yavaş büyüme
        self.life -= 1

    def dead(self): return self.life <= 0

    def draw_to_layer(self, smoke_layer, alpha_layer, h, w):
        """Doğrudan float32 katmana yaz — addWeighted değil, gerçek compositing."""
        t     = 1.0 - self.life / self.ml
        # Önce yüksel sonra sön
        if t < 0.3:
            opacity = t / 0.3 * 0.55
        else:
            opacity = (1.0 - (t-0.3)/0.7) * 0.55

        r = max(4, int(self.r))
        cx, cy = int(self.x), int(self.y)

        # Sınır kontrolü
        x1,x2 = max(0,cx-r), min(w,cx+r)
        y1,y2 = max(0,cy-r), min(h,cy+r)
        if x2<=x1 or y2<=y1: return

        # Gaussian kernel — gerçek yumuşak blob
        rr = x2-x1
        rc = y2-y1
        yg, xg = np.mgrid[0:rc, 0:rr].astype(np.float32)
        dx = xg - (cx-x1)
        dy = yg - (cy-y1)
        g  = np.exp(-(dx**2+dy**2)/(2*(r*0.45)**2)) * opacity

        for c in range(3):
            smoke_layer[y1:y2, x1:x2, c] = np.maximum(
                smoke_layer[y1:y2, x1:x2, c],
                self.color[c] * g
            )
        alpha_layer[y1:y2, x1:x2] = np.maximum(
            alpha_layer[y1:y2, x1:x2], g
        )


def render_steam(frame, particles, lm, intensity):
    if not particles and intensity < 0.01: return frame
    h, w = frame.shape[:2]
    smoke_layer = np.zeros((h, w, 3), np.float32)
    alpha_layer = np.zeros((h, w),    np.float32)

    for p in particles:
        p.draw_to_layer(smoke_layer, alpha_layer, h, w)

    if alpha_layer.max() < 0.001: return frame

    # Hafif blur → daha da yumuşat
    alpha_layer = gaussian_blur_fast(alpha_layer, 15)
    for c in range(3):
        smoke_layer[:,:,c] = gaussian_blur_fast(smoke_layer[:,:,c], 9)

    alpha_layer = np.clip(alpha_layer, 0, 1)
    return alpha_blend(frame, smoke_layer.astype(np.uint8), alpha_layer)


# ═══════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████
#  FRANKY — SHOCKWAVE PATLAMA
#  • Eliptik dalga halkaları + glow
#  • Merkez flaş
#  • Fırlayan enerji çizgileri
# ██████████████████████████████████████████████████════
# ═══════════════════════════════════════════════════════
class ShockRing:
    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w,  self.h  = w,  h
        self.rx    = float(random.randint(10,30))
        self.max_rx= float(random.randint(180,320))
        self.life  = 1.0   # 0..1
        self.speed = random.uniform(0.035, 0.06)
        self.angle = random.uniform(-15,15)
        # Sarı-beyaz-cyan enerji
        r,g,b = random.choice([(255,240,120),(200,240,255),(255,255,255),(150,230,255)])
        self.color = (b,g,r)  # BGR

    def dead(self): return self.life <= 0

    def update(self): self.life -= self.speed

    def draw(self, glow_layer, sharp_layer):
        t    = 1.0 - self.life
        rx   = int(self.rx + (self.max_rx-self.rx)*t)
        ry   = int(rx * 0.50)
        a    = self.life ** 0.6   # hızlı başlar, yavaş söner
        if rx < 2: return

        # Sharp kenarda ince çizgi
        cv2.ellipse(sharp_layer,(self.cx,self.cy),(rx,ry),
                    self.angle,0,360,self.color,2,cv2.LINE_AA)

        # Glow katmanı: kalın + bulanık
        cv2.ellipse(glow_layer,(self.cx,self.cy),(rx,ry),
                    self.angle,0,360,self.color,12,cv2.LINE_AA)
        cv2.ellipse(glow_layer,(self.cx,self.cy),(max(1,rx-6),max(1,ry-3)),
                    self.angle,0,360,self.color,6,cv2.LINE_AA)


class EnergyBeam:
    """Patlamadan fırlayan ışın çizgisi."""
    def __init__(self, cx, cy):
        self.cx,self.cy = cx,cy
        a = random.uniform(0, math.pi*2)
        spd = random.uniform(6,18)
        self.x,  self.y  = float(cx), float(cy)
        self.vx, self.vy = math.cos(a)*spd, math.sin(a)*spd
        self.life  = random.randint(12,28)
        self.ml    = self.life
        self.length= random.randint(15,40)
        r,g,b = random.choice([(255,240,100),(200,240,255),(255,255,200)])
        self.color=(b,g,r)

    def dead(self): return self.life<=0

    def update(self):
        self.x+=self.vx; self.y+=self.vy
        self.vy+=0.4; self.vx*=0.92; self.life-=1

    def draw(self, layer):
        a   = self.life/self.ml
        ex  = int(self.x - self.vx*0.6)
        ey  = int(self.y - self.vy*0.6)
        # Kalın + ince → glow hissi
        cv2.line(layer,(int(self.x),int(self.y)),(ex,ey),self.color,4,cv2.LINE_AA)
        cv2.line(layer,(int(self.x),int(self.y)),(ex,ey),(255,255,255),1,cv2.LINE_AA)


def render_shockwave(frame, rings, beams, cx, cy, flash):
    h, w = frame.shape[:2]

    # Merkez flaş — sadece tetiklendiği an
    if flash > 0.01:
        fl = radial_gradient(cx, cy, int(w*0.35), h, w)
        fl = (fl * flash * 0.6).astype(np.float32)
        white = np.full((h,w,3), 255, np.float32)
        frame = alpha_blend(frame, white.astype(np.uint8), fl)

    if not rings and not beams: return frame

    glow_layer  = np.zeros((h,w,3), np.uint8)
    sharp_layer = np.zeros((h,w,3), np.uint8)

    for r in rings: r.draw(glow_layer, sharp_layer)

    # Glow blur → parıldama
    glow_blur = gaussian_blur_fast(glow_layer, 31)

    # Blend: önce glow, sonra sharp üstüne
    glow_a = glow_blur.astype(np.float32).max(axis=2) / 255.0 * 0.65
    frame  = alpha_blend(frame, glow_blur, glow_a)

    sharp_a = sharp_layer.astype(np.float32).max(axis=2) / 255.0 * 0.90
    frame   = alpha_blend(frame, sharp_layer, sharp_a)

    # Işın çizgileri
    beam_layer = np.zeros((h,w,3), np.uint8)
    for b in beams: b.draw(beam_layer)
    beam_blur = gaussian_blur_fast(beam_layer, 5)
    beam_a    = beam_blur.astype(np.float32).max(axis=2)/255.0 * 0.85
    frame     = alpha_blend(frame, beam_blur, beam_a)

    return frame


# ═══════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████
#  ROBİN — GERÇEKÇİ TEN RENGİ KOLLAR
#  • Bezier eğrisi → doğal kol formu
#  • Omuzdan bileğe incelme + outline → hacim hissi
#  • 5 parmak, eklem detayı, tırnak noktası
#  • Kenar yumuşatma (anti-aliased kalın çizgi)
# ██████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════

SKIN_BASE  = (165, 195, 215)   # BGR — açık ten
SKIN_DARK  = (110, 140, 160)   # gölge / outline
SKIN_LIGHT = (200, 225, 235)   # highlight

class RealisticArm:
    def __init__(self, fw, fh, tx, ty):
        self.fw, self.fh = fw, fh
        self.life     = random.randint(50, 80)
        self.ml       = self.life

        # Kalınlık
        self.sw = random.randint(16, 24)   # shoulder width
        self.ww = random.randint(9,  13)   # wrist width

        # Başlangıç: kenardan
        edge = random.randint(0,3)
        m    = 80
        if   edge==0: sx,sy = random.randint(m,fw-m), -30
        elif edge==1: sx,sy = random.randint(m,fw-m), fh+30
        elif edge==2: sx,sy = -30, random.randint(m,fh-m)
        else:         sx,sy = fw+30, random.randint(m,fh-m)

        # Dirsek kontrol noktası (Bezier orta)
        ex = (sx+tx)//2 + random.randint(-80,80)
        ey = (sy+ty)//2 + random.randint(-70,70)

        self.P0 = np.array([sx,  sy],  float)
        self.P1 = np.array([ex,  ey],  float)
        self.P2 = np.array([tx + random.randint(-25,25),
                             ty + random.randint(-25,25)], float)

        # Parmak yönü: vücuda işaret eder
        self.finger_angle = math.atan2(ty-sy, tx-sx) + random.uniform(-0.6,0.6)

    def dead(self): return self.life<=0

    def _bezier(self, t):
        t = clamp(t,0,1)
        return (1-t)**2*self.P0 + 2*(1-t)*t*self.P1 + t**2*self.P2

    def draw(self, arm_layer, h, w):
        progress = 1.0 - self.life/self.ml
        reach    = clamp(progress*2.0, 0, 1)   # ilk yarıda uzan
        alpha    = clamp(self.life/self.ml, 0, 1)

        steps = 28
        pts   = [self._bezier(i/steps*reach) for i in range(steps+1)]
        ipts  = [(int(p[0]),int(p[1])) for p in pts]

        # Kol gövdesi — outline + skin + highlight
        for i in range(len(ipts)-1):
            t      = i/steps
            w_line = int(self.sw*(1-t) + self.ww*t)
            p1, p2 = ipts[i], ipts[i+1]

            # 1. Koyu outline (hacim kenarı)
            cv2.line(arm_layer, p1, p2, SKIN_DARK,  w_line+5, cv2.LINE_AA)
            # 2. Orta ten
            cv2.line(arm_layer, p1, p2, SKIN_BASE,  w_line,   cv2.LINE_AA)
            # 3. İnce highlight (üstten ışık)
            if w_line > 6:
                offset = max(1, w_line//3)
                # highlight çizgisi hafifçe offset
                dx = p2[1]-p1[1]; dy = p1[0]-p2[0]
                n  = max(math.hypot(dx,dy), 1)
                ox,oy = int(dx/n*offset), int(dy/n*offset)
                cv2.line(arm_layer,
                         (p1[0]+ox,p1[1]+oy),(p2[0]+ox,p2[1]+oy),
                         SKIN_LIGHT, max(1,w_line//3), cv2.LINE_AA)

        # El
        if reach > 0.70 and len(ipts) > 0:
            wx, wy = ipts[-1]
            hr = self.ww + 7

            cv2.circle(arm_layer,(wx,wy), hr+3, SKIN_DARK, -1, cv2.LINE_AA)
            cv2.circle(arm_layer,(wx,wy), hr,   SKIN_BASE, -1, cv2.LINE_AA)
            cv2.circle(arm_layer,(wx,wy), max(1,hr-4), SKIN_LIGHT, 2, cv2.LINE_AA)

            # 5 parmak
            lens   = [16, 21, 23, 20, 14]
            angles = [self.finger_angle + math.radians(-45+j*22) for j in range(5)]
            for fi,(fa,fl) in enumerate(zip(angles,lens)):
                fx = int(wx + math.cos(fa)*fl)
                fy = int(wy + math.sin(fa)*fl)
                mx = int(wx + math.cos(fa)*fl*0.55)
                my = int(wy + math.sin(fa)*fl*0.55)

                # Parmak: outline + skin
                cv2.line(arm_layer,(wx,wy),(fx,fy),SKIN_DARK, 6, cv2.LINE_AA)
                cv2.line(arm_layer,(wx,wy),(fx,fy),SKIN_BASE, 4, cv2.LINE_AA)
                # Eklem
                cv2.circle(arm_layer,(mx,my), 3, SKIN_DARK, -1, cv2.LINE_AA)
                cv2.circle(arm_layer,(mx,my), 2, SKIN_BASE, -1, cv2.LINE_AA)
                # Tırnak ucu
                cv2.circle(arm_layer,(fx,fy), 2, SKIN_DARK, -1, cv2.LINE_AA)

        return alpha

    def draw_to_frame(self, frame):
        h, w = frame.shape[:2]
        arm_layer = np.zeros((h,w,3), np.uint8)
        alpha     = self.draw(arm_layer, h, w)
        self.life -= 1

        # Hafif edge blur → daha doğal geçiş
        arm_blurred = gaussian_blur_fast(arm_layer, 3)
        a_mask = arm_blurred.astype(np.float32).max(axis=2)/255.0 * alpha
        return alpha_blend(frame, arm_blurred, a_mask)


# ═══════════════════════════════════════════════════════
# KARAKTER AURASI
# ═══════════════════════════════════════════════════════
def apply_aura(frame, lm_px, color_bgr, intensity):
    if not lm_px or intensity<0.01: return frame
    h, w = frame.shape[:2]
    pts  = np.array(list(lm_px.values()), np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Dış glow
    dil   = cv2.dilate(mask, np.ones((45,45),np.uint8))
    glow  = gaussian_blur_fast(dil.astype(np.float32)/255.0*intensity, 41)

    color_layer = np.zeros((h,w,3), np.uint8)
    color_layer[:] = color_bgr

    # Çift katman: yumuşak dış + parlak iç
    outer = np.clip(glow*0.55, 0, 1)
    frame = alpha_blend(frame, color_layer, outer)

    inner_mask = gaussian_blur_fast(mask.astype(np.float32)/255.0*intensity*0.25, 21)
    inner      = np.clip(inner_mask, 0, 1)
    frame      = alpha_blend(frame, color_layer, inner)

    return frame


# ═══════════════════════════════════════════════════════
# POZ ALGILAYICILARI (v4'ten aynen)
# ═══════════════════════════════════════════════════════
def score_luffy_gear2(lm):
    try:
        ls,rs=lm["LEFT_SHOULDER"],lm["RIGHT_SHOULDER"]
        le,re=lm["LEFT_ELBOW"],lm["RIGHT_ELBOW"]
        lw,rw=lm["LEFT_WRIST"],lm["RIGHT_WRIST"]
        lh,rh=lm["LEFT_HIP"],lm["RIGHT_HIP"]
        lk,rk=lm["LEFT_KNEE"],lm["RIGHT_KNEE"]
        nose=lm["NOSE"]
    except KeyError: return 0.0
    sw=max(dist(ls,rs),1.0); kw=dist(lk,rk)
    smid=mid(ls,rs); hmid=mid(lh,rh)
    if not (lh[1]>=lk[1]-sw*0.3 and rh[1]>=rk[1]-sw*0.3): return 0.0
    wide=1.0 if kw>sw*1.15 else 0.0
    tfwd=1.0 if smid[1]>=hmid[1]-sw*0.2 else 0.0
    hlow=1.0 if nose[1]>=smid[1]-sw*0.05 else 0.0
    hwlow=1.0 if (lw[1]>=lk[1]-sw*0.4 and rw[1]>=rk[1]-sw*0.4) else 0.0
    bent=1.0 if (angle(ls,le,lw)<145 and angle(rs,re,rw)<145) else 0.0
    return 0.30*wide+0.20*tfwd+0.15*hlow+0.20*hwlow+0.15*bent

def score_franky_super(lm):
    try:
        ls,rs=lm["LEFT_SHOULDER"],lm["RIGHT_SHOULDER"]
        le,re=lm["LEFT_ELBOW"],lm["RIGHT_ELBOW"]
        lw,rw=lm["LEFT_WRIST"],lm["RIGHT_WRIST"]
        lh,rh=lm["LEFT_HIP"],lm["RIGHT_HIP"]
        lk,rk=lm["LEFT_KNEE"],lm["RIGHT_KNEE"]
    except KeyError: return 0.0
    sw=max(dist(ls,rs),1.0); kw=dist(lk,rk)
    smid=mid(ls,rs); hmid=mid(lh,rh)
    lu=lw[1]<ls[1]-sw*0.8; ru=rw[1]<rs[1]-sw*0.8
    if not (lu or ru): return 0.0
    if lu:
        astr=1.0 if angle(ls,le,lw)>155 else 0.4
        odown=1.0 if rw[1]>smid[1]+sw*0.1 else 0.0
    else:
        astr=1.0 if angle(rs,re,rw)>155 else 0.4
        odown=1.0 if lw[1]>smid[1]+sw*0.1 else 0.0
    wide=1.0 if kw>sw*1.2 else 0.0
    tilt=1.0 if abs(ls[1]-rs[1])>sw*0.15 else 0.5
    return 0.35*astr+0.25*odown+0.25*wide+0.15*tilt

def score_robin_fleur(lm):
    try:
        ls,rs=lm["LEFT_SHOULDER"],lm["RIGHT_SHOULDER"]
        le,re=lm["LEFT_ELBOW"],lm["RIGHT_ELBOW"]
        lw,rw=lm["LEFT_WRIST"],lm["RIGHT_WRIST"]
        lh,rh=lm["LEFT_HIP"],lm["RIGHT_HIP"]
    except KeyError: return 0.0
    sw=max(dist(ls,rs),1.0); smid=mid(ls,rs); hmid=mid(lh,rh)
    th=max(hmid[1]-smid[1],1.0)
    cl=smid[1]+th*0.20; ch=smid[1]+th*0.65
    if not (cl<lw[1]<ch and cl<rw[1]<ch): return 0.0
    if not (lw[0]>smid[0] and rw[0]<smid[0]): return 0.0
    close=1.0 if dist(lw,rw)<sw*0.65 else 0.0
    lb=1.0 if 45<angle(ls,le,lw)<145 else 0.0
    rb=1.0 if 45<angle(rs,re,rw)<145 else 0.0
    lf=1.0 if abs(lw[0]-smid[0])<sw*0.8 else 0.0
    rf=1.0 if abs(rw[0]-smid[0])<sw*0.8 else 0.0
    return 0.35*close+0.25*lb+0.25*rb+0.075*lf+0.075*rf

# ═══════════════════════════════════════════════════════
# KARAKTER KONFİG
# ═══════════════════════════════════════════════════════
CHARACTERS = [
    {"id":"luffy_gear","name":"Luffy","move":"Gear 2nd",
     "score_fn":score_luffy_gear2,"threshold":0.60,"hold":10,
     "sound":"luffy_gear","label":"GEAR SECOND!",
     "aura_color":(200,50,160),"text_color":(255,100,255),"effect":"smoke"},
    {"id":"franky","name":"Franky","move":"SUPER",
     "score_fn":score_franky_super,"threshold":0.60,"hold":8,
     "sound":"franky_super","label":"SUPEEEER!",
     "aura_color":(180,120,0),"text_color":(0,220,255),"effect":"shockwave"},
    {"id":"robin","name":"Robin","move":"Treinta Fleur",
     "score_fn":score_robin_fleur,"threshold":0.60,"hold":8,
     "sound":"robin_fleur","label":"Treinta Fleur!",
     "aura_color":(150,20,150),"text_color":(220,100,255),"effect":"arms"},
]

# ═══════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════
def draw_text_glow(frame, text, x, y, scale, color):
    """Işıltılı efektli yazı."""
    # Glow katmanı
    glow_layer = np.zeros_like(frame)
    cv2.putText(glow_layer,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,scale,color,8,cv2.LINE_AA)
    glow_blur = gaussian_blur_fast(glow_layer, 21)
    glow_a    = glow_blur.astype(np.float32).max(axis=2)/255.0*0.7
    frame     = alpha_blend(frame, glow_blur, glow_a)
    # Siyah outline
    cv2.putText(frame,text,(x+2,y+2),cv2.FONT_HERSHEY_DUPLEX,scale,(0,0,0),4,cv2.LINE_AA)
    # Ana yazı
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,scale,color,2,cv2.LINE_AA)
    return frame

def draw_hud(frame, states, w, h):
    bw=180; x0=15; y0=h-15-len(CHARACTERS)*40
    # Yarı saydam arka plan
    overlay = frame.copy()
    cv2.rectangle(overlay,(x0-8,y0-24),(x0+bw+115,y0+len(CHARACTERS)*40-4),(10,10,10),-1)
    cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)

    for i,char in enumerate(CHARACTERS):
        st=states[i]; sc=st["score"]; thr=char["threshold"]; y=y0+i*40
        cv2.putText(frame,f"{char['name']} - {char['move']}",(x0,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(220,220,220),1,cv2.LINE_AA)
        cv2.rectangle(frame,(x0,y+5),(x0+bw,y+17),(40,40,40),-1)
        fill=int(bw*min(sc,1.0))
        col=(0,255,80) if sc>=thr else char["aura_color"]
        cv2.rectangle(frame,(x0,y+5),(x0+fill,y+17),col,-1)
        # Bar glow
        if fill>4:
            bar_ov=frame.copy()
            cv2.rectangle(bar_ov,(x0,y+5),(x0+fill,y+17),col,-1)
            cv2.addWeighted(bar_ov,0.3,frame,0.7,0,frame)
        tx=x0+int(bw*thr)
        cv2.line(frame,(tx,y+3),(tx,y+19),(255,255,255),1)
        cv2.putText(frame,f"{sc:.2f}",(x0+bw+8,y+15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1,cv2.LINE_AA)

def draw_skeleton(frame, raw, w, h):
    CONNS=[(11,12),(11,13),(13,15),(12,14),(14,16),
           (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]
    pts={}
    for a,b in CONNS:
        for idx in(a,b):
            if idx not in pts and idx<len(raw):
                pts[idx]=(int(raw[idx].x*w),int(raw[idx].y*h))
    for a,b in CONNS:
        if a in pts and b in pts:
            cv2.line(frame,pts[a],pts[b],(0,180,0),2,cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(frame,pt,4,(0,255,0),-1,cv2.LINE_AA)

# ═══════════════════════════════════════════════════════
# ANA DÖNGÜ
# ═══════════════════════════════════════════════════════
def main():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
    print("[MP] PoseLandmarker hazır.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit("[HATA] Kamera açılamadı.")

    COOLDOWN = 4.0
    states   = [{"score":0.0,"hold":0,"triggered":False,
                 "last_time":0.0,"aura":0.0} for _ in CHARACTERS]

    steam_particles = []
    shock_rings     = []
    energy_beams    = []
    robin_arms      = []
    flash_val       = 0.0     # Franky merkez flaş
    active_char     = None
    frame_ts        = 0

    print("="*50)
    print("  One Piece x MediaPipe  v6")
    print("  Q = çıkış")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame   = cv2.flip(frame,1)
        h, w, _ = frame.shape
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        result  = landmarker.detect_for_video(mp_img, frame_ts)
        frame_ts += 33

        raw = result.pose_landmarks[0] if result.pose_landmarks else []
        lm  = extract_lm(raw, w, h) if raw else {}
        now = time.time()

        chest = mid(lm.get("LEFT_SHOULDER",(w//2,h//3)),
                    lm.get("RIGHT_SHOULDER",(w//2,h//3)))

        # ── skor & tetikleme ──
        for i, char in enumerate(CHARACTERS):
            st    = states[i]
            score = char["score_fn"](lm) if lm else 0.0
            st["score"] = score
            in_cd = (now - st["last_time"]) < COOLDOWN

            if score >= char["threshold"] and not in_cd:
                st["hold"] += 1
            else:
                st["hold"] = max(0, st["hold"]-1)

            if st["hold"] > char["hold"] and not st["triggered"] and not in_cd:
                st["triggered"] = True
                st["last_time"] = now
                st["aura"]      = 1.0
                active_char     = i
                play_sound(char["sound"])
                print(f"[✓] {char['name']} – {char['label']}")

                if char["effect"] == "shockwave":
                    for _ in range(10):
                        shock_rings.append(ShockRing(chest[0],chest[1],w,h))
                    for _ in range(55):
                        energy_beams.append(EnergyBeam(chest[0],chest[1]))
                    flash_val = 1.0

                elif char["effect"] == "arms":
                    for _ in range(12):
                        robin_arms.append(RealisticArm(w,h,chest[0],chest[1]))

            if not in_cd and st["hold"] == 0:
                st["triggered"] = False

        # ── aura ──
        for i, char in enumerate(CHARACTERS):
            st = states[i]
            if st["aura"] > 0.01:
                pulse = 0.70 + 0.30*math.sin(now*7.0)
                frame = apply_aura(frame, lm, char["aura_color"],
                                   st["aura"]*pulse)
                st["aura"] = max(0.0, st["aura"]-0.007)

                if char["effect"] == "smoke" and lm:
                    # Her karede birden fazla noktadan duman
                    pts_smoke = [
                        lm.get("LEFT_SHOULDER"), lm.get("RIGHT_SHOULDER"),
                        lm.get("LEFT_ELBOW"),    lm.get("RIGHT_ELBOW"),
                        lm.get("LEFT_HIP"),      lm.get("RIGHT_HIP"),
                        lm.get("LEFT_KNEE"),     lm.get("RIGHT_KNEE"),
                    ]
                    for pt in pts_smoke:
                        if pt and random.random() < 0.6:
                            steam_particles.append(SteamPuff(pt[0],pt[1]))

                if char["effect"] == "shockwave" and st["aura"] > 0.15:
                    if random.random() < 0.35:
                        shock_rings.append(ShockRing(chest[0],chest[1],w,h))
                    if random.random() < 0.15:
                        for _ in range(8):
                            energy_beams.append(EnergyBeam(chest[0],chest[1]))

                if char["effect"] == "arms" and st["aura"] > 0.15:
                    if random.random() < 0.20:
                        robin_arms.append(RealisticArm(w,h,chest[0],chest[1]))

        # ════════════════════════════════════════════
        # ÇİZİM SIRASI:
        # 1. Duman (arkada)
        # 2. Shockwave
        # 3. İskelet
        # 4. Robin kollar (önde)
        # 5. HUD & yazı
        # ════════════════════════════════════════════

        # 1. Steam (Luffy)
        for p in steam_particles: p.update()
        steam_particles = [p for p in steam_particles if not p.dead()][-400:]
        frame = render_steam(frame, steam_particles, lm,
                             states[0]["aura"] if states else 0)

        # 2. Shockwave (Franky)
        flash_val = max(0.0, flash_val-0.08)
        for r in shock_rings: r.update()
        shock_rings  = [r for r in shock_rings  if not r.dead()][-30:]
        for b in energy_beams: b.update()
        energy_beams = [b for b in energy_beams if not b.dead()][-150:]
        frame = render_shockwave(frame, shock_rings, energy_beams,
                                 chest[0], chest[1], flash_val)

        # 3. İskelet
        if raw: draw_skeleton(frame, raw, w, h)

        # 4. Robin kollar
        alive=[]
        for a in robin_arms:
            frame = a.draw_to_frame(frame)
            if not a.dead(): alive.append(a)
        robin_arms = alive[-28:]

        # 5. Aktif yazı
        if active_char is not None:
            st = states[active_char]
            if st["aura"] > 0.01:
                char = CHARACTERS[active_char]
                sz   = cv2.getTextSize(char["label"],cv2.FONT_HERSHEY_DUPLEX,1.5,2)[0]
                cx   = (w-sz[0])//2
                frame = draw_text_glow(frame, char["label"], cx, 85, 1.5,
                                       char["text_color"])
            else:
                active_char = None

        # 6. HUD
        draw_hud(frame, states, w, h)

        # 7. Başlık
        cv2.rectangle(frame,(0,0),(w,32),(15,15,15),-1)
        cv2.putText(frame,"ONE PIECE x MediaPipe  |  Q = cikis",
                    (10,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,220,80),1,cv2.LINE_AA)

        cv2.imshow("One Piece – Pose Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()