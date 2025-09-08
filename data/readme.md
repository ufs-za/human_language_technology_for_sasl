# 🤟 SASL HLT Data Folder

This folder contains **annotated and synthetic datasets** for **South African Sign Language Human Language Technology (SASL HLT)**, supporting research and development under the [Interdisciplinary Centre for Digital Futures](https://github.com/ufs-za/Interdisciplinary-Centre-for-Digital-Futures/tree/main?tab=readme-ov-file) at the [University of the Free State](https://www.ufs.ac.za/).

---

## 📁 Structure

```plaintext
/meshes/                     → STL files of SASL fingerspelling meshes, organized by letter
  ├── Aa/                    → Example: SASL_Aa_Right_Medium_var1.stl (variations for hand orientation, size, pose)
  │   └── metadata.json      → Describes the meshes for each letter (e.g., mesh count, variations)
/synthetic_data_resources/   → Synthetic dataset from videos of painted 3D-printed hands
  ├── Aa/                    → Organized by letter and lot number (e.g., FS_AaRM250607#1/)
  │   ├── frames/            → PNG frames (e.g., FS_AaRM250607#1_frame_0001.png)
  │   ├── source_video.mp4   → Original video of the hand
  │   └── metadata.json      → Lot-specific metadata (e.g., lot number, hand details, capture date)
/annotations/                → Metadata for signs
  └── asl_sasl.csv           → Metadata for signs (e.g., name, YouTube link, SASL/ASL similarity, number of hands)
```

---

## 📋 Formats

* **Meshes**: STL
* **Videos**: MP4
* **Frames**: PNG
* **Annotations/Metadata**: CSV, JSON

---

## 🧾 Licensing

All data is governed by the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**  
🔗 [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

You are free to:

* **Share** — copy and redistribute the material in any medium or format
* **Adapt** — remix, transform, and build upon the material

Under the following terms:

* **Attribution** — you must give appropriate credit
* **Non-Commercial** — no commercial use is allowed
* **ShareAlike** — distribute your contributions under the same license

🛑 **Commercial use is prohibited without a formal data sharing agreement.**

See [`/docs/LICENCE.md`](../../docs/LICENCE.md) for full license details.

---

## ✅ Data Use Requirements

* **Attribution** 🤝: Always credit the **University of the Free State** and the **Interdisciplinary Centre for Digital Futures**.
* **Non-Commercial** 🚫: Do not use for commercial purposes without written agreement.
* **ShareAlike** 🔄: Do not redistribute without preserving the **CC BY-NC-SA 4.0** license.
* **Contact** 📬: For commercial or derivative use, email [contact details](https://www.ufs.ac.za/icdf/icdf-home/contact-us).