# 🤟 South African Sign Language Human Language Technology (SASL HLT)

Welcome to the official repository for **South African Sign Language Human Language Technology (SASL HLT)**. All of this work falls within the scope of the [Interdisciplinary Centre for Digital Futures](https://github.com/ufs-za/Interdisciplinary-Centre-for-Digital-Futures/tree/main?tab=readme-ov-file) at the [University of the Free State](https://www.ufs.ac.za/). This initiative promotes open, inclusive, and ethical development of HLT tools, datasets, and models for SASL. The project is non-commercial and maintained by researchers at the **University of the Free State**, guided by Ubuntu ethics and a commitment to linguistic justice.

---

## 🌍 Purpose

This repository is dedicated to the **development, documentation, and open access of SASL HLT** for research, education, and public use. We aim to correct the hearing-centric bias in existing HLT systems by supporting **Deaf-led and inclusive design**. Our work supports community-anchored applications, including the *Lebitso App*, and aligns with principles of linguistic equality and technological equity.

---

## 📁 Repository Structure

```plaintext
/data                     → Annotated and synthetic datasets for SASL HLT
  ├── meshes/             → STL files of SASL fingerspelling meshes, organized by letter (e.g., Aa/, Bb/)
  │   └── metadata.json   → Mesh metadata (e.g., variations, creation method)
  ├── synthetic_data_resources/ → Synthetic video datasets of painted 3D-printed hands, organized by letter and lot (e.g., Aa/FS_AaRM250607#1/)
  │   └── metadata.json   → Lot-specific metadata (e.g., lot number, hand details)
  ├── annotations/        → Sign metadata in CSV format (e.g., asl_sasl.csv)
  └── README.md           → Detailed data folder documentation
/images                   → Visuals, avatars, interface assets
/code                     → Python scripts for processing, modelling, and tool development
/notebooks                → Jupyter notebooks for analysis and experimentation
/docs                     → Licensing, ethical frameworks, project vision
/streamlit                → Streamlit demo apps and UI prototypes
/artificial_intelligence  → AI models, reinforcement learning, synthetic data tools
```

---

## 🧠 Technology Stack

* **Language**: Python
* **Tools**: Jupyter, Streamlit, Pandas, OpenCV, Blender
* **Formats**: CSV, MP4, JSON, PKL, PNG, STL
* **Platforms**: Local, Google Colab, Streamlit Cloud (planned)

---

## ✅ Features

* Annotated datasets and glossing dictionaries
* Synthetic datasets including meshes (STL) and video frames (PNG) for fingerspelling
* Metadata files (JSON) for meshes and synthetic video lots, detailing variations and hand details
* Streamlit applications for interactive demos (e.g., Lebitso App)
* Code and models for preprocessing, analysis, and AI training
* Ethical licensing and data governance documentation
* Notebooks for exploratory research and reproducible pipelines

---

## 🛤️ Roadmap

* [ ] End-to-end SASL recognition and translation pipelines
* [ ] Avatar-based SASL synthesis and variation generation
* [ ] Sign-language-informed glossing engines
* [ ] Benchmarks and metrics for inclusive SASL HLT
* [ ] Real-time APIs and annotation services

---

## 🧾 License and Ethical Use

All content in this repository is released under the:

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**  
🔗 [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

You are free to:

* **Share** — copy and redistribute the material in any medium or format
* **Adapt** — remix, transform, and build upon the material

Under the following terms:

* **Attribution** — you must give appropriate credit
* **Non-Commercial** — no commercial use is allowed
* **ShareAlike** — distribute your contributions under the same license

🛑 **Commercialisation requires a formal data sharing agreement.**  
✅ All users must provide proper acknowledgements when using any part of this repository.

See [`/docs/LICENCE.md`](docs/LICENCE.md) for more details.

---

## 🧭 Vision and Values

This repository forms part of our response to the **HLT infodemic** — where existing technologies for sign language are predominantly **hearing-centric** and often exclude Deaf perspectives. We advocate for:

* **Ubuntu Ethics** — recognising interdependence, dignity, and inclusion
* **Nothing About Us Without Us** — prioritising Deaf community leadership
* **Non-Commercial, Public Good** — for learning, research, and community empowerment

---

## 🤝 Contribution Guidelines

We welcome contributions that align with our ethical and community values.

To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

Please read:

* [`/docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)
* [`/docs/CODE_OF_CONDUCT.md`](docs/CODE_OF_CONDUCT.md)

---

## 🧪 Quickstart

To run locally:

```bash
git clone https://github.com/YourOrg/sasl-hlt.git
cd sasl-hlt
pip install -r requirements.txt
```

To launch a Streamlit app:

```bash
streamlit run streamlit/app_name.py
```

Explore notebooks in `/notebooks` or datasets in `/data` to begin your experiments.

---

## 🔗 Links

* 🌐 Main Website: [https://www.ufs.ac.za/icdf](https://www.ufs.ac.za/icdf)
* 📝 Research: [`/docs/RESEARCH.md`](docs/RESEARCH.md)
* 📊 Streamlit Apps: [`/streamlit/`](streamlit/)
* 🎓 Educational Tools: [`/notebooks/`](notebooks/)
* 🖼️ Image Assets: [`/images/`](images/)

---

## 📬 Contact

For queries, collaborations, or ethical approvals:

* **Email**: [https://www.ufs.ac.za/icdf/icdf-home/contact-us](https://www.ufs.ac.za/icdf/icdf-home/contact-us)
* **Institution**: University of the Free State
* **Affiliation**: Interdisciplinary Centre for Digital Futures