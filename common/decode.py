from core.decoding.decoder import Decoder
from core.decoding.spell import Spell
from core.utils.labels import labels_to_text


DECODER_GREEDY     = True
DECODER_BEAM_WIDTH = 200


def create_decoder(dict_path: str, greedy: bool = DECODER_GREEDY, width: int = DECODER_BEAM_WIDTH):
	spell = Spell(dict_path)
	return Decoder(greedy=greedy, beam_width=width, postprocessors=[labels_to_text, spell.sentence])
