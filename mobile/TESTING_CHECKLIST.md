# NEXUS Mobile App - Testing Checklist

## Pre-Testing Setup

- [ ] Node.js 18+ installed
- [ ] npm dependencies installed (`npm install`)
- [ ] Expo CLI available (`npx expo --version`)
- [ ] iOS Simulator / Android Emulator configured
- [ ] Backend API running (optional, for cloud tests)

## Build Verification

### TypeScript
- [x] `npx tsc --noEmit` passes with no errors
- [x] All imports resolve correctly
- [x] Type definitions are complete

### Expo Config
- [x] app.json schema valid
- [x] Asset files exist (icon, splash, adaptive-icon)
- [ ] Plugins configured correctly (expo-camera, expo-av)

---

## Screen-by-Screen Testing

### 1. Home Screen
| Test | Expected Result | Status |
|------|-----------------|--------|
| App launches | Home screen displays | [ ] |
| Comprehensive assessments visible | Maternal & Newborn cards show | [ ] |
| Quick assessments visible | Anemia, Jaundice, Cry cards show | [ ] |
| Navigation to Maternal Assessment | Opens PregnantWomanScreen | [ ] |
| Navigation to Newborn Assessment | Opens NewbornScreen | [ ] |
| Navigation to Quick Assessments | Opens respective screens | [ ] |

### 2. Maternal Assessment (PregnantWomanScreen)
| Test | Expected Result | Status |
|------|-----------------|--------|
| Patient info form displays | Gestational weeks, gravida, para fields | [ ] |
| Danger signs checklist loads | 8 WHO IMNCI items | [ ] |
| Can select danger signs | Checkboxes toggle correctly | [ ] |
| Camera opens for conjunctiva | Camera permission requested | [ ] |
| Image capture works | Photo saved and displayed | [ ] |
| Analysis runs | Loading state, then results | [ ] |
| Results displayed | Navigates to Results screen | [ ] |

### 3. Newborn Assessment (NewbornScreen)
| Test | Expected Result | Status |
|------|-----------------|--------|
| Patient info form displays | Age, weight, APGAR fields | [ ] |
| Danger signs checklist loads | 10 WHO IMNCI items | [ ] |
| Kramer zone reference shown | Visual guide displays | [ ] |
| Skin photo capture works | Camera opens, photo saved | [ ] |
| Audio recording works | Mic permission, 5-10 sec recording | [ ] |
| Multi-modal analysis runs | Jaundice + Cry analysis | [ ] |
| Combined results displayed | All findings on Results screen | [ ] |

### 4. Quick Anemia Screen
| Test | Expected Result | Status |
|------|-----------------|--------|
| Instructions displayed | How to capture conjunctiva | [ ] |
| Camera opens | Permission granted | [ ] |
| Image picker works | Gallery selection works | [ ] |
| Edge AI toggle works | Switches inference mode | [ ] |
| Analysis completes | Results displayed | [ ] |

### 5. Quick Jaundice Screen
| Test | Expected Result | Status |
|------|-----------------|--------|
| Instructions displayed | Kramer zone reference | [ ] |
| Camera opens | Permission granted | [ ] |
| Analysis completes | Severity, bilirubin estimate | [ ] |

### 6. Quick Cry Analysis Screen
| Test | Expected Result | Status |
|------|-----------------|--------|
| Instructions displayed | Recording guidance | [ ] |
| Record button works | Starts/stops recording | [ ] |
| Timer displays | Shows recording duration | [ ] |
| Playback works | Can play recorded audio | [ ] |
| Analysis completes | Asphyxia risk displayed | [ ] |

### 7. Results Screen
| Test | Expected Result | Status |
|------|-----------------|--------|
| Anemia results display | Probability, hemoglobin, risk | [ ] |
| Jaundice results display | Severity, bilirubin, Kramer zone | [ ] |
| Cry results display | Risk score, cry type | [ ] |
| Clinical synthesis shows | MedGemma recommendations | [ ] |
| WHO classification shown | RED/YELLOW/GREEN badge | [ ] |
| Danger signs listed | Selected signs displayed | [ ] |

---

## Offline Capability Testing

### Network State
| Test | Expected Result | Status |
|------|-----------------|--------|
| Online indicator shows | Green status when connected | [ ] |
| Offline indicator shows | Red/yellow when disconnected | [ ] |
| Network status updates | Automatic detection | [ ] |

### Local Database
| Test | Expected Result | Status |
|------|-----------------|--------|
| Patient saved locally | SQLite insert works | [ ] |
| Assessment saved locally | SQLite insert works | [ ] |
| Data persists after app restart | Records still present | [ ] |
| Query patients works | List retrieves correctly | [ ] |

### Sync Queue
| Test | Expected Result | Status |
|------|-----------------|--------|
| Pending items tracked | Count shows correctly | [ ] |
| Sync executes when online | Items processed | [ ] |
| Failed items retry | Exponential backoff works | [ ] |
| Sync progress shown | Progress bar updates | [ ] |

### Edge AI (When Available)
| Test | Expected Result | Status |
|------|-----------------|--------|
| Edge toggle available | Switch in UI | [ ] |
| Local inference runs | No network call | [ ] |
| Results match cloud | Similar accuracy | [ ] |

---

## Error Handling Testing

### Error Boundary
| Test | Expected Result | Status |
|------|-----------------|--------|
| Crash caught | Error UI displayed | [ ] |
| Retry button works | App recovers | [ ] |
| Error details shown (dev) | Stack trace in dev mode | [ ] |

### Toast Notifications
| Test | Expected Result | Status |
|------|-----------------|--------|
| Success toast shows | Green toast | [ ] |
| Error toast shows | Red toast | [ ] |
| Warning toast shows | Yellow toast | [ ] |
| Auto-dismiss works | Toast disappears | [ ] |

### Loading States
| Test | Expected Result | Status |
|------|-----------------|--------|
| Loading overlay shows | During analysis | [ ] |
| AI animation plays | Pulse effect | [ ] |
| Cancel possible | If long-running | [ ] |

---

## UI/UX Testing

### Visual Design
| Test | Expected Result | Status |
|------|-----------------|--------|
| Colors consistent | Brand colors used | [ ] |
| Fonts readable | Proper sizing | [ ] |
| Icons display | Ionicons render | [ ] |
| Cards styled correctly | Shadows, borders | [ ] |

### Responsiveness
| Test | Expected Result | Status |
|------|-----------------|--------|
| Portrait orientation | Layout correct | [ ] |
| Different screen sizes | Scales properly | [ ] |
| Scroll works | Long content scrollable | [ ] |
| Keyboard handling | Inputs accessible | [ ] |

### Accessibility
| Test | Expected Result | Status |
|------|-----------------|--------|
| Touch targets adequate | 44pt minimum | [ ] |
| Color contrast sufficient | WCAG AA | [ ] |
| Screen reader labels | Accessible text | [ ] |

---

## Integration Testing

### API Integration
| Test | Expected Result | Status |
|------|-----------------|--------|
| Anemia API call works | Response received | [ ] |
| Jaundice API call works | Response received | [ ] |
| Cry API call works | Response received | [ ] |
| Synthesis API call works | Combined results | [ ] |
| API error handled | User-friendly message | [ ] |

### Agentic Workflow
| Test | Expected Result | Status |
|------|-----------------|--------|
| Triage agent runs | Risk score generated | [ ] |
| Image agent runs | MedSigLIP results | [ ] |
| Audio agent runs | HeAR results | [ ] |
| Protocol agent runs | WHO classification | [ ] |
| Referral agent runs | Final recommendation | [ ] |

---

## Performance Testing

### Startup Time
| Test | Target | Status |
|------|--------|--------|
| Cold start | < 3 seconds | [ ] |
| Warm start | < 1 second | [ ] |

### Analysis Time
| Test | Target | Status |
|------|--------|--------|
| Image analysis (cloud) | < 5 seconds | [ ] |
| Audio analysis (cloud) | < 5 seconds | [ ] |
| Combined analysis | < 10 seconds | [ ] |
| Edge inference | < 2 seconds | [ ] |

### Memory Usage
| Test | Target | Status |
|------|--------|--------|
| Idle memory | < 100 MB | [ ] |
| During analysis | < 300 MB | [ ] |
| No memory leaks | Stable over time | [ ] |

---

## Device Testing Matrix

| Device | OS Version | Status |
|--------|------------|--------|
| iPhone 15 Pro | iOS 17 | [ ] |
| iPhone SE | iOS 16 | [ ] |
| Pixel 7 | Android 14 | [ ] |
| Samsung A53 | Android 13 | [ ] |
| Low-end Android | Android 10 | [ ] |

---

## Demo Preparation

### Demo Data
- [ ] Sample conjunctiva images ready
- [ ] Sample skin images ready
- [ ] Sample cry audio files ready
- [ ] Mock patient data loaded

### Demo Flow
- [ ] Maternal assessment demo tested
- [ ] Newborn assessment demo tested
- [ ] Quick assessments demo tested
- [ ] Offline mode demo tested
- [ ] Results screen demo tested

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| QA Tester | | | |
| Product Owner | | | |

---

*Last Updated: January 14, 2026*
*Version: 1.0*
