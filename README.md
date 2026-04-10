# AstroSearchLight

Vision-based light curve classification.

## Models

### SimuLens

ResNet trained to jointly classify segmented light curves. Light curves are split into segments of 2 days, 7 days, and infinite length (i.e. full light curve). Segments are plotted as scatter plots and the CNN is trained to classify the plots.

Trained on simulated TESS data:

- (Self-)Lensing: 10,297
- Stellar flares: 10,043
- Not-lensing (including transits): 9,012

### HandLens (planned)

To be trained on manual (Zooniverse) labels of real light curves.

### FalseLens (planned)

To be trained on manual (Zooniverse) vetting/false positive removal with CCD/pixel data.

### Deprecated

- XGBoost
- Upsilon