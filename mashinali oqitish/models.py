import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

# ==============================================================================
# 1. MANGA BUBBLE & TEXT DETECTOR (Object Detection)
# ==============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MangaBubbleDetector(nn.Module):
    """
    Object Detector for Manga Speech Bubbles, Narration Boxes, and Sound Effects (SFX).
    Predicts bounding boxes (cx, cy, w, h) and class logits.
    """
    def __init__(self, num_classes=4, num_anchors=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone (Feature Extractor)
        self.enc1 = ConvBlock(3, 32, stride=2)   # H/2, W/2
        self.enc2 = ConvBlock(32, 64, stride=2)  # H/4, W/4
        self.enc3 = ConvBlock(64, 128, stride=2) # H/8, W/8
        self.enc4 = ConvBlock(128, 256, stride=2) # H/16, W/16
        
        # Feature Aggregation / Detection Neck
        self.neck = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Head: (4 bbox coords + 1 objectness + num_classes) * num_anchors
        out_channels = num_anchors * (5 + num_classes)
        self.head = nn.Conv2d(128, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        x: (B, 3, H, W)
        Returns:
            predictions: (B, num_anchors, H/16, W/16, 5 + num_classes)
        """
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        features = self.neck(f4)
        
        preds = self.head(features)
        B, C, H, W = preds.shape
        preds = preds.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        preds = preds.permute(0, 1, 3, 4, 2).contiguous() # (B, A, H, W, 5+C)
        return preds

    def predict(self, x, conf_thresh=0.35, iou_thresh=0.45):
        """
        Inference: decodes grid predictions, converts to absolute coordinates,
        and applies Non-Maximum Suppression (NMS).
        Args:
            x: (1, 3, H, W) normalized tensor
        Returns:
            list of dicts: [{'box': [x1, y1, x2, y2], 'score': float, 'class_id': int}, ...]
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)  # (1, num_anchors, Gh, Gw, 5 + C)
            B, A, Gh, Gw, D = preds.shape
            preds = preds.view(-1, D)
            
            shift_y, shift_x = torch.meshgrid(
                torch.arange(0, Gh, device=x.device, dtype=torch.float32),
                torch.arange(0, Gw, device=x.device, dtype=torch.float32),
                indexing="ij"
            )
            grid = torch.stack([shift_x, shift_y], dim=-1).view(-1, 2)
            
            obj_scores = torch.sigmoid(preds[:, 4])
            cls_probs = torch.softmax(preds[:, 5:], dim=-1)
            cls_scores, cls_indices = torch.max(cls_probs, dim=-1)
            scores = obj_scores * cls_scores
            
            mask = scores > conf_thresh
            if not mask.any():
                return []
                
            filtered_preds = preds[mask]
            filtered_scores = scores[mask]
            filtered_cls = cls_indices[mask]
            filtered_grid = grid[mask]
            
            pred_xy = (torch.sigmoid(filtered_preds[:, :2]) + filtered_grid) / torch.tensor([Gw, Gh], device=x.device)
            pred_wh = torch.sigmoid(filtered_preds[:, 2:4])
            
            x1 = torch.clamp(pred_xy[:, 0] - pred_wh[:, 0] / 2.0, 0.0, 1.0)
            y1 = torch.clamp(pred_xy[:, 1] - pred_wh[:, 1] / 2.0, 0.0, 1.0)
            x2 = torch.clamp(pred_xy[:, 0] + pred_wh[:, 0] / 2.0, 0.0, 1.0)
            y2 = torch.clamp(pred_xy[:, 1] + pred_wh[:, 1] / 2.0, 0.0, 1.0)
            x2 = torch.maximum(x1 + 1e-4, x2)
            y2 = torch.maximum(y1 + 1e-4, y2)
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            keep = nms(boxes, filtered_scores, iou_thresh)
            
            results = []
            for idx in keep:
                results.append({
                    "box": boxes[idx].cpu().tolist(),
                    "score": float(filtered_scores[idx].item()),
                    "class_id": int(filtered_cls[idx].item())
                })
            return results

# ==============================================================================
# 2. MANGA OCR MODEL (Vision + BiLSTM + CTC Loss)
# ==============================================================================

class MangaOCRModel(nn.Module):
    """
    Hybrid Deep CNN + BiLSTM Architecture for Multi-Language Manga OCR.
    Supports both:
    1. Line-level sequence OCR with BiLSTM + CTC Loss on text line crops (e.g. 64x256).
       Output: (T, B, vocab_size) log_softmax
    2. Character-level CNN classification on single character crops (e.g. 64x64).
       Output: (B, vocab_size) logits
    """
    def __init__(self, vocab_size=3516, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Deep Convolutional Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32x32 (or 32x128 for sequence)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 16x16 (or 16x64 for sequence)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 8x8 (or 8x32 for sequence)
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 1. Sequence Head (BiLSTM + Linear for CTC)
        self.seq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.1
        )
        self.seq_fc = nn.Linear(hidden_dim * 2, vocab_size)
        
        # 2. Character Head (Single character 64x64 classifier)
        self.char_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 4 * 4, hidden_dim if hidden_dim >= 256 else 256),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim if hidden_dim >= 256 else 256, vocab_size)
        )

    def forward(self, x, return_sequence=None):
        """
        x: (B, 3, H, W)
        If return_sequence is True or (return_sequence is None and x.shape[-1] > 64):
            Returns sequence log_probs for CTC loss: (T, B, vocab_size)
        Else:
            Returns character logits for CrossEntropyLoss: (B, vocab_size)
        """
        feats = self.features(x)
        is_seq = return_sequence if return_sequence is not None else (x.shape[-1] > 64)
        
        if is_seq:
            # feats: (B, 128, H_out, W_out) -> collapse H to 1
            pooled = self.seq_pool(feats).squeeze(2) # (B, 128, W_out)
            seq_in = pooled.permute(2, 0, 1)         # (T, B, 128)
            recurrent, _ = self.rnn(seq_in)          # (T, B, hidden_dim * 2)
            logits = self.seq_fc(recurrent)          # (T, B, vocab_size)
            return F.log_softmax(logits, dim=-1)
        else:
            pooled = self.char_pool(feats)           # (B, 128, 4, 4)
            logits = self.fc(pooled)                 # (B, vocab_size)
            return logits

    def load_checkpoint_weights(self, state_dict):
        """
        Robust checkpoint weight loader that handles vocabulary size changes,
        preserving existing trained character weights without shape mismatch crashes.
        Handles both character classification head (fc) and sequence CTC head (seq_fc).
        """
        state = dict(state_dict)
        fc_w_key = "fc.4.weight" if "fc.4.weight" in state else ("fc.weight" if "fc.weight" in state else None)
        old_w = None
        old_b = None
        if fc_w_key and state[fc_w_key].shape[0] != self.vocab_size:
            old_w = state.pop(fc_w_key)
            fc_b_key = fc_w_key.replace("weight", "bias")
            if fc_b_key in state:
                old_b = state.pop(fc_b_key)

        old_seq_w = None
        old_seq_b = None
        if "seq_fc.weight" in state and state["seq_fc.weight"].shape[0] != self.vocab_size:
            old_seq_w = state.pop("seq_fc.weight")
            if "seq_fc.bias" in state:
                old_seq_b = state.pop("seq_fc.bias")

        # Expanding a vocabulary may initialize new rows, but never replace
        # missing weights for existing classes with a random model bias.
        if old_w is not None and old_b is None:
            raise ValueError("Incomplete OCR checkpoint: missing character head bias")
        if old_seq_w is not None and old_seq_b is None:
            raise ValueError("Incomplete OCR checkpoint: missing sequence head bias")

        allowed_missing = set()
        if old_w is not None:
            allowed_missing.update({"fc.4.weight", "fc.4.bias"})
        if old_seq_w is not None:
            allowed_missing.update({"seq_fc.weight", "seq_fc.bias"})
        incompatible = self.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys) - allowed_missing
        if missing or incompatible.unexpected_keys:
            raise ValueError(f"Incomplete OCR checkpoint: missing={sorted(missing)}, unexpected={incompatible.unexpected_keys}")

        with torch.no_grad():
            if old_w is not None and hasattr(self, "fc") and len(self.fc) >= 5:
                min_c = min(old_w.shape[0], self.vocab_size)
                self.fc[4].weight[:min_c] = old_w[:min_c]
                if old_b is not None:
                    self.fc[4].bias[:min_c] = old_b[:min_c]

            if old_seq_w is not None and hasattr(self, "seq_fc"):
                min_s = min(old_seq_w.shape[0], self.vocab_size)
                self.seq_fc.weight[:min_s] = old_seq_w[:min_s]
                if old_seq_b is not None:
                    self.seq_fc.bias[:min_s] = old_seq_b[:min_s]

# ==============================================================================
# 3. MANGA INPAINTING MODEL (Deep U-Net for Text & Bubble Removal)
# ==============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class MangaInpaintingModel(nn.Module):
    """
    Deep U-Net Architecture for Manga Text Removal & Screentone Restoration.
    Takes 4-channel input: [RGB (3 channels) + Binary Inpainting Mask (1 channel)]
    Produces 3-channel restored clean image.
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Dilated Bottleneck (for large receptive field to capture screentone patterns)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        # Final output layer
        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # Pixel values in [0, 1]
        )
        
    def forward(self, image, mask):
        """
        image: (B, 3, H, W) in [0, 1]
        mask:  (B, 1, H, W) binary (1 = inpaint/remove, 0 = keep)
        Returns:
            infilled_image: (B, 3, H, W)
            composite_image: (B, 3, H, W) - blending original outside mask with infilled inside mask
        """
        # Ensure mask is strictly 1-channel
        if mask.shape[1] > 1:
            mask = mask[:, :1]

        # Ensure spatial dimensions are divisible by 8 for U-Net down/up sampling
        orig_h, orig_w = image.shape[-2], image.shape[-1]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            image_pad = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
            mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            image_pad = image
            mask_pad = mask

        # Mask out text region from input image so skip connections cannot leak text
        masked_image = image_pad * (1.0 - mask_pad)
        x = torch.cat([masked_image, mask_pad], dim=1) # (B, 4, H, W)
        
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decode
        u1 = self.up1(b)
        c1 = self.conv_up1(torch.cat([u1, x3], dim=1))
        
        u2 = self.up2(c1)
        c2 = self.conv_up2(torch.cat([u2, x2], dim=1))
        
        u3 = self.up3(c2)
        c3 = self.conv_up3(torch.cat([u3, x1], dim=1))
        
        out = self.outc(c3)
        
        # Unpad back to original dimensions if padded
        if pad_h > 0 or pad_w > 0:
            out = out[..., :orig_h, :orig_w]

        # Composite: output matches original on unmasked pixels, infilled on masked pixels
        composite = image * (1.0 - mask) + out * mask
        return out, composite

if __name__ == "__main__":
    torch.manual_seed(42)
    print("Testing Model Instantiations...")
    # 1. Detection
    det_model = MangaBubbleDetector(num_classes=4)
    dummy_img = torch.randn(2, 3, 384, 256)
    det_out = det_model(dummy_img)
    preds = det_model.predict(dummy_img[:1])
    print("MangaBubbleDetector output shape:", det_out.shape, "| Predict results:", len(preds))
    
    # 2. OCR
    ocr_model = MangaOCRModel(vocab_size=1500)
    dummy_crop = torch.randn(2, 3, 64, 256)
    ocr_out = ocr_model(dummy_crop)
    print("MangaOCRModel output shape:", ocr_out.shape)
    
    # 3. Inpainting
    inpaint_model = MangaInpaintingModel()
    dummy_mask = torch.ones(2, 1, 256, 256)
    dummy_inpaint_img = torch.randn(2, 3, 256, 256)
    raw_out, comp_out = inpaint_model(dummy_inpaint_img, dummy_mask)
    print("MangaInpaintingModel output shape:", comp_out.shape)
    print("All models initialized and verified successfully!")
