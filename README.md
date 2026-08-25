# 📚 TheMoodShelf

**TheMoodShelf** is a wellness-driven book recommendation system that connects human emotions with meaningful reading experiences.

🔗 **Live Demo:**  
https://resincinnamon-themoodshelf.hf.space  

---

## 🧠 Overview

TheMoodShelf allows users to express their feelings in natural language and receive book recommendations that align with their emotional state.

Instead of searching manually, users can simply type:
- A word  
- A sentence  
- A feeling  

The system detects the mood using NLP models and suggests relevant books.

---

## ✨ Features

- 🎭 **Mood Detection using NLP**
  - Utilizes **DistilRoBERTa** and **MiniLM**
  - Understands emotions from free-text input

- 📖 **Personalized Book Recommendations**
  - Suggests books based on detected mood
  - Incorporates **user history** for improved personalization

- 🌐 **Web Application**
  - Simple and clean UI using HTML & CSS

- ⚡ **Fast Backend**
  - Built with Flask for efficient processing

- 🗄️ **Database Integration**
  - Uses MySQL for storing user data and book records

---

## 🏗️ Tech Stack

- **Frontend:** HTML, CSS  
- **Backend:** Flask (Python)  
- **NLP Models:** DistilRoBERTa, MiniLM  
- **Database:** MySQL (FreeSQLDatabase)  
- **Deployment:** Hugging Face Spaces (Docker-based)

---

## 🚀 How It Works

1. User inputs a feeling (e.g., *"I feel anxious and overwhelmed"*)
2. Text is processed using NLP models:
   - DistilRoBERTa for sentiment understanding  
   - MiniLM for semantic similarity  
3. Mood is classified into categories
4. System checks **user’s past interactions** for personalization
5. Relevant books are fetched from the MySQL database
6. Recommendations are displayed to the user

---

## 🌍 Deployment

- Hosted on **Hugging Face Spaces**
- Uses a **Docker-based deployment pipeline**
- Database hosted on **FreeSQLDatabase**

---

## ⚠️ Important Notes

- 🗄️ **Database Limitation:**  
  The project uses a free-tier MySQL database (**FreeSQLDatabase**) which has a storage limit of **5MB**. This may restrict the number of stored records.

- ⏳ **Cold Start Delay:**  
  The Hugging Face Space may take some time to load if it has been inactive (cold start), especially on the free tier.

---

## 📌 Use Cases

- 📘 Emotion-based book discovery  
- 🧘 Wellness and self-care tools  
- 🎓 NLP & recommendation system projects  
- 💡 Demonstration of sentiment-aware applications  

---

## 🔮 Future Improvements

- Advanced recommendation algorithms  
- Larger and more diverse book dataset  
- Real-time user analytics  
- Integration with external APIs (Google Books, Goodreads)  
- Improved scalability beyond free-tier limitations  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.
