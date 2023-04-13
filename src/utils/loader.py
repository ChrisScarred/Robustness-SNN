from config import Config

def load_config() -> Config:
	from dotenv import load_dotenv
	env = load_dotenv()
	config_src = env.get("CONFIG_PATH", "config/example.cfg")
	return Config(config_src)
