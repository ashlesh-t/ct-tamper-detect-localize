# config/theme.py
def load_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Roboto+Mono&display=swap');

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
    }
    .title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem !important;
        background: linear-gradient(90deg, #00ff9d, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 0 0 30px rgba(0, 255, 157, 0.5);
        animation: glow 3s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 20px #00ff9d; }
        to { text-shadow: 0 0 40px #00d4ff, 0 0 60px #00ff9d; }
    }
    .subtitle {
        font-family: 'Roboto Mono', monospace;
        color: #94a3b8;
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 3rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00ff9d, #00d4ff);
        color: black;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        box-shadow: 0 0 20px #00ff9d;
    }
    .coords {
        position: fixed;
        top: 120px;
        right: 20px;
        background: rgba(0,0,0,0.8);
        padding: 10px 15px;
        border-radius: 10px;
        color: #00ff9d;
        font-family: 'Orbitron';
        border: 1px solid #00ff9d;
        z-index: 999;
    }
.stSlider > div > div > div > div {
    background: linear-gradient(to right, #00ff9d, #00c8ff) !important;
    height: 20px !important;
    border-radius: 4px !important;
    border: 2px solid #00ff9d !important;
}
.stSlider > div > div > div {
    background: #111 !important;
    border-radius: 6px !important;
    height: 24px !important;
}
    </style>
    """