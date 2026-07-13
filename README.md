# PSDN

PSDN is a two-stage progressive denoising framework for unsupervised visible-infrared person re-identification (US-VI-ReID). The ALS module is designed for global soft label correction, while the SMS module performs local boundary refinement. The two modules work together to suppress pseudo-label noise and boost cross-modal feature alignment performance.

The task of this paper is unsupervised visible-infrared person re-identification (US-VI-ReID). Given visible and infrared pedestrian images, the model achieves cross-modal pedestrian identity matching and retrieval without any manual annotations. The proposed PSDN framework adopts TransReID ViT as the feature extraction backbone. It learns modality-invariant discriminative features via global label smoothing and local boundary refinement, alleviates pseudo-label noise caused by clustering, and improves the stability of cross-modal feature alignment as well as retrieval accuracy.

## overview
