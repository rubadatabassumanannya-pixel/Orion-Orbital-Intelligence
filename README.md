# ORION Orbital Intelligence 🛰️

**Live orbital risk intelligence platform for the Asia-Pacific region.**

🔗 **Live Platform:** [orion-orbital-intelligence.netlify.app](https://orion-orbital-intelligence.netlify.app)

---

## What is ORION?

ORION is a space situational awareness (SSA) platform built for satellite operators, university CubeSat programs, and emerging space agencies across South and Southeast Asia — a region severely underserved by existing SSA solutions.

It tells operators not just the risk, but what to do about it.

---

## Technical Stack

- **Orbital Propagation:** SGP4 (same standard used by NASA and NORAD)
- **Conjunction Screening:** KDTree spatial indexing across active satellite catalog
- **Probability of Collision:** Foster method with encounter-plane Gaussian projection
- **Covariance Modelling:** RTN-frame TLE-age covariance (Radial, Transverse, Normal)
- **False Alarm Simulation:** 200,000-encounter Monte Carlo study
- **Frontend:** JavaScript, HTML, CSS — deployed on Netlify
- **Data Source:** CelesTrak TLE catalog

---

## Key Features

- Real-time conjunction alerts with mathematically computed Pc values
- TLE-age covariance modelling — uncertainty grows with data age
- Data provenance on every alert — method, source, TLE age, uncertainty class
- Space weather monitoring — solar flux, Kp index, drag index
- AI mission assistant for decision support
- Dynamic time-based greetings and 72-hour mission timeline
- Historical replay — 2009 Iridium-Cosmos collision
- Location search with globe fly-to animation

---

## Original Research Finding

Running a 200,000-encounter Monte Carlo simulation, ORION demonstrated that:

- The standard 10⁻⁴ Pc threshold produces **zero alerts** with TLE-grade data
- Custom thresholds calibrated to TLE uncertainty are required for operational screening
- Pc-based alerting gives fewer false alarms than raw miss distance at matched recall

---

## Files

| File | Description |
|------|-------------|
| `index.html` | Full ORION dashboard — globe, alerts, AI assistant, space weather |
| `orion_screen.py` | Python conjunction screening pipeline — SGP4, Pc, covariance |
| `orion_falsealarm.py` | 200,000-encounter false alarm comparison simulation |
| `conjunctions.json` | Generated real conjunction data with Pc and TLE age |

---

## About

Built independently by **Rubada Tabassum** — aerospace engineer, Springer-published researcher (ICSATI 2025), based in Dhaka, Bangladesh.

Built without a team, without a lab, without a university — between work calls and late nights.

---

## Contact

📧 tabassumrubada90@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/your-linkedin-here)  
🛰️ [orion-orbital-intelligence.netlify.app](https://orion-orbital-intelligence.netlify.app)
