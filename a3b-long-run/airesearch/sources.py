"""Channel sources for the R1 registry (docs/PLAN.md sec 4).

Three provenances:
  seed       - owner's quality bar (PLAN 4.1)
  candidate  - PLAN 4.2 hand list
  expansion  - PLAN 4.3 topic sweeps / channel graph
Each entry is a YouTube channel URL. ``name`` is only a label; the fetch step
records the real channel title from yt-dlp.
"""

SEEDS = [
    ("IBM Technology", "https://www.youtube.com/@IBMTechnology"),
    ("AI Search", "https://www.youtube.com/channel/UCIgnGlGkVRhd4qNFcEwLL4A"),
    ("Aperture", "https://www.youtube.com/channel/UCO5QSoES5yn2Dw7YixDYT5Q"),
    ("3Blue1Brown", "https://www.youtube.com/@3blue1brown"),
    ("Perimeter Institute", "https://www.youtube.com/channel/UCpHvNclapYpq8b0ZnGRoMeg"),
    ("neuro-dump", "https://www.youtube.com/channel/UCMHBJH5rX2dF1S7o-fJuY-A"),
]

CANDIDATES = [
    # paper explainers
    ("Yannic Kilcher", "https://www.youtube.com/@YannicKilcher"),
    ("Two Minute Papers", "https://www.youtube.com/@TwoMinutePapers"),
    ("bycloud", "https://www.youtube.com/@bycloudAI"),
    ("AI Coffee Break", "https://www.youtube.com/@AICoffeeBreak"),
    ("Gabriel Mongaras", "https://www.youtube.com/@gabrielmongaras"),
    ("hu-po", "https://www.youtube.com/@hu-po"),
    ("Tunadorable", "https://www.youtube.com/channel/UCfFCdHblZylZfxnHDFPKnPQ"),
    ("Julia Turc", "https://www.youtube.com/channel/UCMUy_WsUIYbM_hKR9drb9dA"),
    ("Algorithmic Simplicity", "https://www.youtube.com/@AlgorithmicSimplicity"),
    # engineering / training
    ("Andrej Karpathy", "https://www.youtube.com/@AndrejKarpathy"),
    ("Umar Jamil", "https://www.youtube.com/@umarjamilai"),
    ("Trelis Research", "https://www.youtube.com/@TrelisResearch"),
    ("Sam Witteveen", "https://www.youtube.com/@samwitteveenai"),
    ("sentdex", "https://www.youtube.com/@sentdex"),
    ("Hamel Husain", "https://www.youtube.com/channel/UC__dUuqF5w4OnbW221JxmKg"),
    ("fast.ai", "https://www.youtube.com/@howardjeremyp"),
    ("Unsloth", "https://www.youtube.com/channel/UCg6e-8DXa2l_HY4co82cMmg"),
    ("Evan Armstrong", "https://www.youtube.com/channel/UCM6Fc142x5uZxtUmRl_wWoA"),
    # intuition / maths
    ("Welch Labs", "https://www.youtube.com/@WelchLabs"),
    ("StatQuest", "https://www.youtube.com/@statquest"),
    ("Serrano.Academy", "https://www.youtube.com/@SerranoAcademy"),
    ("Mutual Information", "https://www.youtube.com/@MutualInformation"),
    ("Steve Brunton", "https://www.youtube.com/@SteveBrunton"),
    ("Artem Kirsanov", "https://www.youtube.com/@ArtemKirsanov"),
    ("Computerphile", "https://www.youtube.com/@Computerphile"),
    # lectures
    ("Stanford Online", "https://www.youtube.com/@stanfordonline"),
    ("MIT OpenCourseWare", "https://www.youtube.com/@mitocw"),
    ("Simons Institute", "https://www.youtube.com/@SimonsInstitute"),
    ("IPAM", "https://www.youtube.com/@IPAMUCLA"),
    # interviews / frontier context
    ("Machine Learning Street Talk", "https://www.youtube.com/@MachineLearningStreetTalk"),
    ("Dwarkesh Patel", "https://www.youtube.com/@DwarkeshPatel"),
    ("Latent Space", "https://www.youtube.com/@LatentSpacePod"),
    ("The Cognitive Revolution", "https://www.youtube.com/@CognitiveRevolution"),
    ("Gradient Dissent", "https://www.youtube.com/@GradientDissent"),
    ("TWIML", "https://www.youtube.com/@TWIMLAI"),
    # news level (high recall)
    ("AI Explained", "https://www.youtube.com/channel/UCNJ1Ymd5yFuUPtn21xtRbbw"),
    ("Matthew Berman", "https://www.youtube.com/@matthew_berman"),
    ("Wes Roth", "https://www.youtube.com/@WesRoth"),
    ("Fireship", "https://www.youtube.com/@Fireship"),
    ("1littlecoder", "https://www.youtube.com/@1littlecoder"),
]

# PLAN 4.3 expansion: channels found via topic sweeps + the paper/channel graph.
# Provenance kept as "expansion" so the delivered registry can show its origin.
EXPANSION = [
    ("DeepLearning.AI", "https://www.youtube.com/@Deeplearningai"),
    ("Hugging Face", "https://www.youtube.com/@HuggingFace"),
    ("Weights & Biases", "https://www.youtube.com/@WeightsBiases"),
    ("Lightning AI", "https://www.youtube.com/channel/UC8m-y0yAFJpX0hRvxH8wJVw"),
    ("PyTorch", "https://www.youtube.com/@PyTorch"),
    ("TensorFlow", "https://www.youtube.com/@TensorFlow"),
    ("Google DeepMind", "https://www.youtube.com/@GoogleDeepMind"),
    ("OpenAI", "https://www.youtube.com/@OpenAI"),
    ("Anthropic", "https://www.youtube.com/@anthropic-ai"),
    ("AI at Meta", "https://www.youtube.com/@AIatMeta"),
    ("Microsoft Research", "https://www.youtube.com/@MicrosoftResearch"),
    ("NVIDIA Developer", "https://www.youtube.com/@NVIDIADeveloper"),
    ("Modular", "https://www.youtube.com/@modularinc"),
    ("Cohere", "https://www.youtube.com/@Cohere"),
    ("LangChain", "https://www.youtube.com/@LangChain"),
    ("LlamaIndex", "https://www.youtube.com/@LlamaIndex"),
    ("Neel Nanda", "https://www.youtube.com/@NeelNanda"),
    ("Robert Miles AI Safety", "https://www.youtube.com/@RobertMilesAI"),
    ("MLOps.community", "https://www.youtube.com/@MLOpsCommunity"),
    ("DataTalksClub", "https://www.youtube.com/@DataTalksClub"),
    ("AI Engineer", "https://www.youtube.com/@aiDotEngineer"),
    ("Lex Fridman", "https://www.youtube.com/@lexfridman"),
    ("No Priors", "https://www.youtube.com/@NoPriorsPodcast"),
    ("Lightcone Podcast", "https://www.youtube.com/@LightconePodcast"),
    ("Eye on AI", "https://www.youtube.com/@EyeOnAI"),
    ("AssemblyAI", "https://www.youtube.com/@AssemblyAI"),
    ("Kaggle", "https://www.youtube.com/@Kaggle"),
    ("AI Jason", "https://www.youtube.com/@AIJasonZ"),
    ("James Briggs", "https://www.youtube.com/@jamesbriggs"),
    ("Krish Naik", "https://www.youtube.com/@krishnaik06"),
    ("Nicholas Renotte", "https://www.youtube.com/@NicholasRenotte"),
    ("Ritvikmath", "https://www.youtube.com/@ritvikmath"),
    ("ML Explained", "https://www.youtube.com/@MLExplained"),
    ("David Shapiro", "https://www.youtube.com/@DaveShap"),
    ("Emergent Garden", "https://www.youtube.com/@EmergentGarden"),
    ("The Alan Turing Institute", "https://www.youtube.com/@AlanTuringInstitute"),
    ("Y Combinator", "https://www.youtube.com/@ycombinator"),
    ("a16z", "https://www.youtube.com/@a16z"),
    ("Super Data Science", "https://www.youtube.com/@SuperDataScience"),
]

def all_sources():
    """Yield (name, url, provenance) tuples; seeds first."""
    for name, url in SEEDS:
        yield name, url, "seed"
    for name, url in CANDIDATES:
        yield name, url, "candidate"
    for name, url in EXPANSION:
        yield name, url, "expansion"
