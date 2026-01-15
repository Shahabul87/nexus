# NEXUS Demo Video Recording Checklist

## Pre-Recording Setup

### Equipment
- [ ] Screen recording software ready (OBS/QuickTime/ScreenFlow)
- [ ] External microphone tested
- [ ] iPhone/Android device with Expo Go installed
- [ ] Good lighting setup
- [ ] Quiet recording environment

### Software
- [ ] Mobile app running in Expo Go: `npx expo start`
- [ ] Backend API running: `uvicorn api.main:app`
- [ ] Streamlit demo running: `streamlit run src/demo/streamlit_app.py`
- [ ] All test images loaded on device

### Demo Assets Location
```
submission/video/assets/
├── anemia/
│   └── *_palpebral.png    # Conjunctiva samples
├── jaundice/
│   ├── sample_jaundice_1.jpg
│   └── sample_jaundice_2.jpg
└── cry/
    └── Real_Infantcry.wav
```

---

## Recording Checklist (3-Minute Video)

### Scene 1: The Hook (0:00-0:25)
- [ ] Statistics overlay ready
- [ ] NEXUS logo transition prepared
- [ ] Narration: "Every day, 800 mothers and 7,400 newborns die..."

### Scene 2: Anemia Demo (0:25-0:45)
- [ ] Open NEXUS app
- [ ] Navigate to "Pregnant Woman Assessment"
- [ ] Show conjunctiva image capture
- [ ] Display analysis animation
- [ ] Show result: "Moderate anemia detected. Recommend iron supplementation."
- [ ] Show 95% accuracy badge

### Scene 3: Cry Analysis Demo (0:45-1:15)
- [ ] Navigate to "Newborn Assessment"
- [ ] Show audio recording interface
- [ ] Record/play cry sample
- [ ] Show HeAR processing animation
- [ ] Display result: "Warning - abnormal cry pattern detected"
- [ ] Show RED alert and referral recommendation

### Scene 4: Jaundice Demo (1:15-1:40)
- [ ] Show skin photo capture
- [ ] Display MedSigLIP analysis
- [ ] Show result: "Elevated bilirubin detected"
- [ ] Show YELLOW severity indicator

### Scene 5: Agentic Workflow (1:40-2:05)
- [ ] Show architecture diagram animation
- [ ] Highlight 5 agents: Triage → Image → Audio → Protocol → Referral
- [ ] Show MedGemma synthesis

### Scene 6: Offline Demo (2:05-2:25)
- [ ] Enable Airplane Mode
- [ ] Run analysis again
- [ ] Show "Works offline on $100 phone"

### Scene 7: Impact Statistics (2:25-2:50)
- [ ] 6.9 million CHWs statistic
- [ ] 295,000 maternal deaths
- [ ] 2.5 million neonatal deaths
- [ ] "250,000 lives saved" animation

### Scene 8: Closing (2:50-3:00)
- [ ] Show baby with mother (stock footage)
- [ ] NEXUS logo
- [ ] Team names

---

## Recording Commands

### Start Development Servers
```bash
# Terminal 1: Mobile app
cd mobile && npx expo start

# Terminal 2: Backend API
cd /Users/mdshahabulalam/myAIDev/kaggleCompetition/MedGemmaImpactChallenge
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 3: Streamlit demo
streamlit run src/demo/streamlit_app.py
```

### Screen Recording Tips
1. Use 1080p resolution minimum
2. Record at 30fps
3. Use consistent window sizes
4. Disable notifications on all devices
5. Clear browser history/cache
6. Use a clean phone home screen

### Audio Tips
1. Record narration separately for better quality
2. Use script from DEMO_VIDEO_SCRIPT.md
3. Keep pace steady (not too fast)
4. Emphasize key statistics

---

## Post-Recording Checklist

- [ ] Review all footage for errors
- [ ] Sync narration with visuals
- [ ] Add captions/subtitles
- [ ] Add background music (optional, low volume)
- [ ] Export at 1080p minimum
- [ ] Upload to YouTube/Vimeo (unlisted)
- [ ] Test video link works

---

## Emergency Fallback

If live app demo fails:
1. Use pre-recorded screen captures
2. Use Streamlit demo as backup
3. Focus on architecture diagrams and impact

---

*Last Updated: January 14, 2026*
