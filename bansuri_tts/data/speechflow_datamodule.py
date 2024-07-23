from typing import Any, Optional, List

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from astravani.data.dataset import TTSDataset, AudioDataset
from astravani.data.tokenizers import CharacterTokenizer

_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_numbers = '0123456789'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ" + '̃' 
_letters_all = ['Â', 'Ã', 'â', 'ā', 't̺', '³', '×', '´', '½', '(', '←', '☞', 'µ', '￼', ']', 'ˆ', '₦', '&', ':', '∩', '!', '●', '—', '¯', '\\', '¤', '⇒', '‘', '±', '°', '℅', ',', ')', '₤', '№', '£', '٭', '²', '|', '″', '«', '₣', '⅛', '†', '₡', '¢', '=', "'", '»', '¿', '₵', '-', '≡', '₿', '₢', '%', '❀', 'Ⓡ', '٪', '‡', '¸', '‽', '¥', '₠', '₥', '¼', '؛', '•', '/', '…', '₫', '$', '₨', '¶', '۔۔۔', '‰', '¬', '↓', '¹', '█', '¨', '٬', '>', '→', '<', '_', 'ʼ', '“', '₪', '⅞', '©', '¡', '॥', '₺', '₩', '€', '₧', '·', '–', '۔', '*', '⅝', '≤', '~', '₹', 'ª', '[', '”', '`', '#', '↵', 'º', '?', '§', '}', '⅜', '@', '{', '′', '−', '¾', '↑', ';', '+', '^', '"', '？', '،', '٫', '₽', '؟', '।', '.', '¦', '®', '™', 'ऀ', 'ँ', 'ं', 'ः', 'ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऌ', 'ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'ऺ', 'ऻ', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ', '्', 'ॎ', 'ॏ', 'ॐ', '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़', 'ॠ', 'ॡ', 'ॢ', 'ॣ', '।', '॥', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९', '॰', 'ॱ', 'ॲ', 'ॳ', 'ॴ', 'ॵ', 'ॶ', 'ॷ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॽ', 'ॾ', 'ॿ', 'ঀ', 'ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'ঌ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', '়', 'ঽ', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ', 'ৗ', 'ড়', 'ঢ়', 'য়', 'ৠ', 'ৡ', 'ৢ', 'ৣ', '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯', 'ৰ', 'ৱ', '৲', '৳', '৴', '৵', '৶', '৷', '৸', '৹', '৺', '৻', 'ৼ', '৽', '৾', 'ਁ', 'ਂ', 'ਃ', 'ਅ', 'ਆ', 'ਇ', 'ਈ', 'ਉ', 'ਊ', 'ਏ', 'ਐ', 'ਓ', 'ਔ', 'ਕ', 'ਖ', 'ਗ', 'ਘ', 'ਙ', 'ਚ', 'ਛ', 'ਜ', 'ਝ', 'ਞ', 'ਟ', 'ਠ', 'ਡ', 'ਢ', 'ਣ', 'ਤ', 'ਥ', 'ਦ', 'ਧ', 'ਨ', 'ਪ', 'ਫ', 'ਬ', 'ਭ', 'ਮ', 'ਯ', 'ਰ', 'ਲ', 'ਲ਼', 'ਵ', 'ਸ਼', 'ਸ', 'ਹ', '਼', 'ਾ', 'ਿ', 'ੀ', 'ੁ', 'ੂ', 'ੇ', 'ੈ', 'ੋ', 'ੌ', '੍', 'ੑ', 'ਖ਼', 'ਗ਼', 'ਜ਼', 'ੜ', 'ਫ਼', '੦', '੧', '੨', '੩', '੪', '੫', '੬', '੭', '੮', '੯', 'ੰ', 'ੱ', 'ੲ', 'ੳ', 'ੴ', 'ੵ', '੶', 'ઁ', 'ં', 'ઃ', 'અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'ઋ', 'ઌ', 'ઍ', 'એ', 'ઐ', 'ઑ', 'ઓ', 'ઔ', 'ક', 'ખ', 'ગ', 'ઘ', 'ઙ', 'ચ', 'છ', 'જ', 'ઝ', 'ઞ', 'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 'ત', 'થ', 'દ', 'ધ', 'ન', 'પ', 'ફ', 'બ', 'ભ', 'મ', 'ય', 'ર', 'લ', 'ળ', 'વ', 'શ', 'ષ', 'સ', 'હ', '઼', 'ઽ', 'ા', 'િ', 'ી', 'ુ', 'ૂ', 'ૃ', 'ૄ', 'ૅ', 'ે', 'ૈ', 'ૉ', 'ો', 'ૌ', '્', 'ૐ', 'ૠ', 'ૡ', 'ૢ', 'ૣ', '૦', '૧', '૨', '૩', '૪', '૫', '૬', '૭', '૮', '૯', '૰', '૱', 'ૹ', 'ૺ', 'ૻ', 'ૼ', '૽', '૾', '૿', 'ଁ', 'ଂ', 'ଃ', 'ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ଌ', 'ଏ', 'ଐ', 'ଓ', 'ଔ', 'କ', 'ଖ', 'ଗ', 'ଘ', 'ଙ', 'ଚ', 'ଛ', 'ଜ', 'ଝ', 'ଞ', 'ଟ', 'ଠ', 'ଡ', 'ଢ', 'ଣ', 'ତ', 'ଥ', 'ଦ', 'ଧ', 'ନ', 'ପ', 'ଫ', 'ବ', 'ଭ', 'ମ', 'ଯ', 'ର', 'ଲ', 'ଳ', 'ଵ', 'ଶ', 'ଷ', 'ସ', 'ହ', '଼', 'ଽ', 'ା', 'ି', 'ୀ', 'ୁ', 'ୂ', 'ୃ', 'ୄ', 'େ', 'ୈ', 'ୋ', 'ୌ', '୍', '୕', 'ୖ', 'ୗ', 'ଡ଼', 'ଢ଼', 'ୟ', 'ୠ', 'ୡ', 'ୢ', 'ୣ', '୦', '୧', '୨', '୩', '୪', '୫', '୬', '୭', '୮', '୯', '୰', 'ୱ', '୲', '୳', '୴', '୵', '୶', '୷', 'ஂ', 'ஃ', 'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 'க', 'ங', 'ச', 'ஜ', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ன', 'ப', 'ம', 'ய', 'ர', 'ற', 'ல', 'ள', 'ழ', 'வ', 'ஶ', 'ஷ', 'ஸ', 'ஹ', 'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', '்', 'ௐ', 'ௗ', '௦', '௧', '௨', '௩', '௪', '௫', '௬', '௭', '௮', '௯', '௰', '௱', '௲', '௳', '௴', '௵', '௶', '௷', '௸', '௹', '௺', 'ఀ', 'ఁ', 'ం', 'ః', 'ఄ', 'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఌ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'క', 'ఖ', 'గ', 'ఘ', 'ఙ', 'చ', 'ఛ', 'జ', 'ఝ', 'ఞ', 'ట', 'ఠ', 'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ', 'న', 'ప', 'ఫ', 'బ', 'భ', 'మ', 'య', 'ర', 'ఱ', 'ల', 'ళ', 'ఴ', 'వ', 'శ', 'ష', 'స', 'హ', 'ఽ', 'ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', '్', 'ౕ', 'ౖ', 'ౘ', 'ౙ', 'ౚ', 'ౠ', 'ౡ', 'ౢ', 'ౣ', '౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯', '౷', '౸', '౹', '౺', '౻', '౼', '౽', '౾', '౿', 'ಀ', 'ಁ', 'ಂ', 'ಃ', '಄', 'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ', 'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ', '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ', 'ೠ', 'ೡ', 'ೢ', 'ೣ', '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', 'ೱ', 'ೲ', 'ഀ', 'ഁ', 'ം', 'ഃ', 'ഄ', 'അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'ഋ', 'ഌ', 'എ', 'ഏ', 'ഐ', 'ഒ', 'ഓ', 'ഔ', 'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ', 'ത', 'ഥ', 'ദ', 'ധ', 'ന', 'ഩ', 'പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'യ', 'ര', 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ഺ', '഻', '഼', 'ഽ', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'ൃ', 'ൄ', 'െ', 'േ', 'ൈ', 'ൊ', 'ോ', 'ൌ', '്', 'ൎ', '൏', 'ൔ', 'ൕ', 'ൖ', 'ൗ', '൘', '൙', '൚', '൛', '൜', '൝', '൞', 'ൟ', 'ൠ', 'ൡ', 'ൢ', 'ൣ', '൦', '൧', '൨', '൩', '൪', '൫', '൬', '൭', '൮', '൯', '൰', '൱', '൲', '൳', '൴', '൵', '൶', '൷', '൸', '൹', 'ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ', 'ൿ', 'ᰀ', 'ᰁ', 'ᰂ', 'ᰃ', 'ᰄ', 'ᰅ', 'ᰆ', 'ᰇ', 'ᰈ', 'ᰉ', 'ᰊ', 'ᰋ', 'ᰌ', 'ᰍ', 'ᰎ', 'ᰏ', 'ᰐ', 'ᰑ', 'ᰒ', 'ᰓ', 'ᰔ', 'ᰕ', 'ᰖ', 'ᰗ', 'ᰘ', 'ᰙ', 'ᰚ', 'ᰛ', 'ᰜ', 'ᰝ', 'ᰞ', 'ᰟ', 'ᰠ', 'ᰡ', 'ᰢ', 'ᰣ', 'ᰤ', 'ᰥ', 'ᰦ', 'ᰧ', 'ᰨ', 'ᰩ', 'ᰪ', 'ᰫ', 'ᰬ', 'ᰭ', 'ᰮ', 'ᰯ', 'ᰰ', 'ᰱ', 'ᰲ', 'ᰳ', 'ᰴ', 'ᰵ', 'ᰶ', '᰷', '᰻', '᰼', '᰽', '᰾', '᰿', '᱀', '᱁', '᱂', '᱃', '᱄', '᱅', '᱆', '᱇', '᱈', '᱉', 'ᱍ', 'ᱎ', 'ᱏ']
_letters_all = list(sorted(set(_letters_all)))

# Export all symbols:
symbols = list(_letters) + list(_letters_all) + list(_letters_ipa) + list(_numbers)



class SpeechFlowDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        train_manifest: str | List[str],
        val_manifest: str | List[str],
        test_manifest: str | List[str],
        sample_rate: int = 24_000,
        min_duration: float = 0.58,
        max_duration: float = 5.0,
        slice_audio: bool = True,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        tokenizer = CharacterTokenizer(symbols)
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = TTSDataset(tokenizer=tokenizer, manifest_fpaths=self.hparams.train_manifest, sample_rate=self.hparams.sample_rate, min_duration=self.hparams.min_duration, max_duration=self.hparams.max_duration, sup_data_types=["mel_spec"], sup_data_path="/home/tts/ttsteam/repos/bansuri-tts/logs")
            self.data_val = TTSDataset(tokenizer=tokenizer, manifest_fpaths=self.hparams.val_manifest, sample_rate=self.hparams.sample_rate, min_duration=self.hparams.min_duration, max_duration=self.hparams.max_duration, sup_data_types=["mel_spec"], sup_data_path="/home/tts/ttsteam/repos/bansuri-tts/logs")
            self.data_test =TTSDataset(tokenizer=tokenizer, manifest_fpaths=self.hparams.test_manifest, sample_rate=self.hparams.sample_rate, min_duration=self.hparams.min_duration, max_duration=self.hparams.max_duration, sup_data_types=["mel_spec"], sup_data_path="/home/tts/ttsteam/repos/bansuri-tts/logs")


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_val.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_test.collate_fn,
        )

if __name__ == "__main__":
    _ = SpeechFlowDataModule()
