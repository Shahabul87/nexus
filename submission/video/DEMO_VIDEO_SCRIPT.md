# NEXUS Demo Video Script

**Target Duration**: 3 minutes
**Format**: Screen recording with voiceover

---

## Video Outline

| Section | Duration | Content |
|---------|----------|---------|
| Opening Hook | 15 sec | Statistics, emotional impact |
| Problem Statement | 30 sec | The maternal-neonatal crisis |
| Solution Overview | 30 sec | NEXUS platform introduction |
| Live Demo - Maternal | 45 sec | Pregnant woman assessment |
| Live Demo - Newborn | 45 sec | Newborn assessment |
| Technical Deep Dive | 30 sec | HAI-DEF models, agentic workflow |
| Closing | 15 sec | Impact potential, call to action |

---

## Detailed Script

### OPENING HOOK (0:00 - 0:15)

**[VISUAL: Dramatic statistics appearing on screen]**

**VOICEOVER:**
> "Every day, 800 women die from pregnancy complications. 7,400 newborns don't survive their first month. 94% of these deaths happen where doctors are scarce, but smartphones are everywhere."

**[VISUAL: World map highlighting affected regions]**

---

### PROBLEM STATEMENT (0:15 - 0:45)

**[VISUAL: Community health worker in rural setting]**

**VOICEOVER:**
> "Meet the 6.9 million community health workers who serve as the only healthcare providers for billions of people. They have dedication, but they lack diagnostic tools."

**[VISUAL: Three conditions with icons]**

> "Maternal anemia affects 40% of pregnant women globally - but detecting it requires blood tests that aren't available. Neonatal jaundice affects over a million babies yearly - delayed detection causes permanent brain damage. Birth asphyxia kills 900,000 newborns - yet early warning signs go unrecognized."

**[VISUAL: Question mark transitioning to NEXUS logo]**

> "What if every health worker could have specialist-level diagnostics... in their pocket?"

---

### SOLUTION OVERVIEW (0:45 - 1:15)

**[VISUAL: NEXUS app opening on phone]**

**VOICEOVER:**
> "Introducing NEXUS - an AI-powered platform that transforms any smartphone into a diagnostic powerhouse."

**[VISUAL: Three icons appearing - Eye, Baby, Sound wave]**

> "Using Google's Health AI Developer Foundations, NEXUS can:
> - Detect maternal anemia from a photo of the inner eyelid
> - Identify neonatal jaundice from skin images
> - Screen for birth asphyxia by analyzing infant cries"

**[VISUAL: Phone in airplane mode icon]**

> "And it works completely offline - because connectivity shouldn't determine survival."

---

### LIVE DEMO - MATERNAL ASSESSMENT (1:15 - 2:00)

**[VISUAL: Screen recording of mobile app]**

**VOICEOVER:**
> "Let me show you how NEXUS works. A health worker opens the app and selects 'Maternal Assessment.'"

**[ACTION: Tap Maternal Assessment card]**

> "First, they enter basic patient information - gestational weeks, number of previous pregnancies."

**[ACTION: Fill in patient info, tap Next]**

> "Next, the WHO IMNCI danger signs checklist. These are the warning signs that indicate immediate danger."

**[ACTION: Show danger signs checklist, select a few]**

> "For anemia screening, they simply take a photo of the inner eyelid - the conjunctiva."

**[ACTION: Camera view, take photo]**

> "MedSigLIP analyzes the image using zero-shot classification. No training data from this patient needed."

**[ACTION: Show analysis in progress, then results]**

> "Within seconds, NEXUS provides:
> - Anemia risk score
> - Estimated hemoglobin level
> - WHO classification: GREEN, YELLOW, or RED
> - Specific recommendations and referral guidance"

**[VISUAL: Results screen with clinical synthesis]**

---

### LIVE DEMO - NEWBORN ASSESSMENT (2:00 - 2:45)

**[VISUAL: Continue screen recording]**

**VOICEOVER:**
> "Now let's assess a newborn. The health worker enters age in hours, birth weight, and APGAR score."

**[ACTION: Navigate to Newborn Assessment, fill info]**

> "The newborn danger signs checklist covers critical conditions like breathing problems, feeding issues, and temperature abnormalities."

**[ACTION: Show danger signs checklist]**

> "For jaundice, a skin photo is analyzed against Kramer zones - the clinical standard for visual assessment."

**[ACTION: Take skin photo, show analysis]**

> "Here's where it gets powerful. The health worker records 5 seconds of the baby's cry."

**[ACTION: Record cry audio]**

> "HeAR - Google's health acoustic model - analyzes the cry pattern for signs of birth asphyxia."

**[ACTION: Show cry analysis results]**

> "All findings are synthesized by MedGemma into a unified clinical recommendation."

**[VISUAL: Final results with agentic workflow visualization]**

---

### TECHNICAL DEEP DIVE (2:45 - 3:15)

**[VISUAL: Architecture diagram animation]**

**VOICEOVER:**
> "Under the hood, NEXUS uses a 5-agent agentic workflow:
> - Triage Agent for initial risk scoring
> - Image Agent powered by MedSigLIP
> - Audio Agent using HeAR embeddings
> - Protocol Agent applying WHO guidelines
> - Referral Agent synthesizing all findings"

**[VISUAL: HAI-DEF model logos]**

> "All three HAI-DEF models work together: MedSigLIP for images, HeAR for audio, and MedGemma for clinical reasoning."

**[VISUAL: INT8 badge, offline icon]**

> "Models are INT8 quantized for edge deployment - running on $100 Android phones without internet."

---

### CLOSING (3:15 - 3:30)

**[VISUAL: Montage of mothers and babies, health workers using phones]**

**VOICEOVER:**
> "NEXUS isn't just technology - it's a lifeline. 6.9 million health workers. Billions of patient encounters. Lives saved."

**[VISUAL: NEXUS logo with tagline]**

> "Every mother deserves a specialist. Every baby deserves a chance. NEXUS makes it possible."

**[VISUAL: MedGemma Impact Challenge logo]**

> "Built with Google HAI-DEF for the MedGemma Impact Challenge."

---

## Production Notes

### Recording Setup
- **Screen Recording**: Use OBS or QuickTime for iOS simulator recording
- **Resolution**: 1920x1080 (16:9)
- **Frame Rate**: 30fps
- **Audio**: Record voiceover separately for cleaner audio

### Visual Assets Needed
1. Opening statistics overlay
2. World map highlighting low-resource regions
3. HAI-DEF model icons (MedSigLIP, HeAR, MedGemma)
4. Architecture diagram animation
5. WHO IMNCI classification colors (RED/YELLOW/GREEN)
6. NEXUS logo and tagline
7. Closing montage images (royalty-free healthcare images)

### Demo Data Preparation
1. Pre-load sample conjunctiva image
2. Pre-load sample newborn skin image
3. Prepare sample cry audio file
4. Configure app with realistic mock data

### Music
- Subtle, hopeful background music
- Fade during voiceover sections
- Crescendo at closing

### Timing Markers
```
0:00 - Opening hook
0:15 - Problem statement
0:45 - Solution overview
1:15 - Maternal demo start
2:00 - Newborn demo start
2:45 - Technical deep dive
3:15 - Closing
3:30 - END
```

---

## Key Messages to Emphasize

1. **All 3 HAI-DEF models used meaningfully**
   - MedSigLIP: Image analysis
   - HeAR: Cry analysis
   - MedGemma: Clinical synthesis

2. **Edge AI capability**
   - Works offline
   - INT8 quantized
   - Low-end device compatible

3. **Agentic workflow**
   - 5 specialized agents
   - Orchestrated decision making
   - WHO protocol integration

4. **Real-world impact**
   - Addresses 295,000+ maternal deaths
   - Addresses 2.4M+ neonatal deaths
   - Scales to 6.9M health workers

---

## Post-Production Checklist

- [ ] Record all screen sections
- [ ] Record voiceover (clear, professional)
- [ ] Add statistics overlays
- [ ] Add architecture animation
- [ ] Add background music
- [ ] Color correction
- [ ] Export at 1080p
- [ ] Add captions/subtitles
- [ ] Review for timing (must be under 3:30)
- [ ] Test playback quality

---

*Script Version: 1.0*
*Last Updated: January 14, 2026*
