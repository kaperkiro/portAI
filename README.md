# portAI

Small CLI for AI-assisted market analysis using Gemini, yfinance, and Alpaca.

It can:

- run a daily market scan
- run the scheduler loop

## Requirements

- Python 3
- a Gemini API key
- Alpaca keys only if you want to run the scheduler/trading flow

## Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
```

2. Activate it:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key_here

# Only needed for the scheduler / Alpaca trading flow
alpaca_key=your_alpaca_key_here
alpaca_secret_key=your_alpaca_secret_here
```

## How To Run

If you already created the local `venv`, you can run commands directly with:

```bash
venv/bin/python main.py <command>
```

Or, after activating the environment:

```bash
python main.py <command>
```

## Commands

Show help:

```bash
venv/bin/python main.py --help
```

Run the one-off daily market analysis:

```bash
venv/bin/python main.py
```

or:

```bash
venv/bin/python main.py daily-market-analysis
```

Start the scheduler:

```bash
venv/bin/python main.py scheduler
```

## Notes

- One-off analysis commands need `GEMINI_API_KEY`.
- The scheduler needs both Gemini and Alpaca credentials.
