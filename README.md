# RainWise - Rainfall Probability Predictor

Umbrella or sunglasses? A **logistic-regression model trained live in your browser** on 3 years of daily weather, with an honest calibration curve.

**Live demo:** https://akash991833.github.io/rainfall-predictor/

## Features

- **Real ML pipeline** - logistic regression trained by gradient descent on 1,095 simulated days; training time measured and displayed
- **Probability gauge** - animated SVG arc with plain-English verdicts and practical advice
- **Factor contributions** - each input's push on the rain log-odds, visualized
- **Calibration curve** - "when the model says 70%, does it rain 70% of the time?" with per-bin sample counts, plus Brier score and accuracy measured on the data
- **Monsoon-aware** - season selector models seasonal rainfall regimes
- **PWA** - installable, offline after first load
- **Premium UI** - white professional design, navy ink, blue accent, dark + light themes, responsive desktop and mobile

## Screenshots

![Desktop - light](screenshots/desktop-light.png)
![Mobile - dark](screenshots/mobile-dark.png)

## Honest limits

Open weather APIs require keys, so training data is **simulated** from realistic climatology (labeled in-app). The model, calibration measurement, and Brier score are real. This is a teaching-grade model - check a real forecast before canceling plans.

## Tech

Vanilla HTML/CSS/JS, zero dependencies.

## Run locally

Open `index.html` or serve the folder (`python3 -m http.server`).
