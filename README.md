﻿# IsPopify

Spotify Songs Classification Dashboard using Streamlit

---

**BM7 - Group 1**

- Segador, John Russel C.
- Tejada, Kurt Dhaniel A.
- Agor, John Schlieden A.

---

### 🔗 Links:

- 🌐 [Streamlit Link](https://ispopify.streamlit.app/)
- 📗 [Google Colab Notebook](https://colab.research.google.com/drive/1If4bvcHwluVCvMzF2pkskTiaiYIFTlqv?usp=sharing)

### 📊 Dataset:

- [30000 Spotify Songs Dataset (Kaggle)](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)

### 💡 Findings / Insights

**1. Dataset Characteristics**

- Among the audio features, `energy` and `loudness` show the strongest positive correlation, indicating that high-energy songs are generally louder
- The genre distribution is balanced, providing equal representation of each genre, which helps the model learn fairly without bias.
- `Pop` and `Latin` genres have the highest average popularity, while EDM has the lowest. Subgenres also show differences, with `Post-Teen Pop` being the most popular and `Progressive Electro House` the least.
- Each genre shows unique characteristics. EDM tracks have the highest energy, while R&B songs are generally less intense. Rap is more speech-focused, while EDM has fewer vocals. Song speed and length also vary, with EDM tracks being faster and Rock songs tending to be longer.

**2. Popularity Level Classification**

- Using Random Forest Classification, the model achieved **_`accuracy of 75.46%`_**, indicating a that model performed relatively well in predicting popularity levels.
- The model achieved a **_precision of 0.77_** and **_recall of 0.76_** for `low-popularity` tracks.
- For `medium-popularity` tracks, the model achieved a **_precision of 0.69_** and **_recall of 0.72_**.
- The model performs best with `high-popularity` tracks, with a **_precision of 0.80_** and **_recall of 0.78_**, suggesting that it can identify most high-popularity tracks effectively.

**3. Genre Classification**

- The genre classification model achieved an **_`accuracy of 58.65%`_**, which indicates that the model has some ability to classify genres, though with room for improvement.
- The `Rock` genre Achieves the highest precision and recall, with **_`0.77`_** and **_`0.80_**`, respectively. This suggests that the model is most effective in classifying rock tracks.
- For `EDM` and `Rap` genres, the model achieved a **_0.66 precision_** and **_0.72 recall_** for EDM, and **_0.59 precision_**
  and **_0.64 recall_** for Rap.
- The model struggles most with `Latin` and `Pop` genres, with lower precision and recall of **_0.40_** for Pop, and **_0.51_** and **_0.47_** for Latin.
- For `R&B` genre, the model achieved a **_precision of 0.54_** and **_recall of 0.47_**.
