import tensorflow as tf
import albumentations as A
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class CarabidModel():
    def __init__(self):
        self._model = tf.keras.models.load_model("model-saves/multi-aug-classifiers/carabid/CARABID-ENSEMBLE-AUG-CLASSIFIER/classifier/savefile.hdf5")
        self._gray_aug = A.Compose([A.ToGray(p=1.0)])
        self._blur_aug = A.Compose([A.Blur(p=1.0)])
        self.standard_img = np.zeros((299, 299, 3))
        self.blurred_img = np.zeros((299, 299, 3))
        self.grayed_img = np.zeros((299, 299, 3))
        self.initialise_labels()
        self.load_feature_mappers()
        self.load_inception_model()
        self.first_pass()

    
    def load_inception_model(self):
        """
        Loads an Inception classifier.
        """
        full_model = tf.keras.models.load_model(f"../model-saves/extractors/carabid/CARABID-EXTRACTOR/extractor/savefile.hdf5")
        self._inception_model = tf.keras.Sequential([full_model.layers[0].layers[0], 
                                                        full_model.layers[0].layers[-1],
                                                        full_model.layers[-1]])

    def load_feature_mappers(self):
        """
        Loads three extractors used to plot the feature outputs from the first convolutional
        layer of the aug classifier.
        """
        dataset_name = "carabid"
        standard_model_path = f"../model-saves/extractors/{dataset_name}/{dataset_name.upper()}-EXTRACTOR/extractor/savefile.hdf5"
        standard_extractor = tf.keras.models.load_model(standard_model_path).layers[0].layers[-1]
        gray_model_path = f"../model-saves/extractors/{dataset_name}/{dataset_name.upper()}-GRAY-EXTRACTOR/extractor/savefile.hdf5"
        gray_extractor = tf.keras.models.load_model(gray_model_path).layers[0].layers[-1]
        blur_model_path = f"../model-saves/extractors/{dataset_name}/{dataset_name.upper()}-BLUR-EXTRACTOR/extractor/savefile.hdf5"
        blur_extractor = tf.keras.models.load_model(blur_model_path).layers[0].layers[-1]
        self._standard_model = tf.keras.Model(inputs=standard_extractor.input, outputs=standard_extractor.layers[1].output)
        self._gray_model = tf.keras.Model(inputs=gray_extractor.input, outputs=gray_extractor.layers[1].output)
        self._blur_model = tf.keras.Model(inputs=blur_extractor.input, outputs=blur_extractor.layers[1].output)

    def generate_feature_maps(self):
        """
        Plots the feature maps from the first convolutional layer of each extractor in the aug model.
        """
        standard_features = self._standard_model.predict(np.expand_dims(preprocess_input(np.copy(self.standard_img)), axis=0))
        blur_features = self._blur_model.predict(np.expand_dims(preprocess_input(np.copy(self.blurred_img)), axis=0))
        gray_features = self._gray_model.predict(np.expand_dims(preprocess_input(np.copy(self.grayed_img)), axis=0))
        features = []
        for i in range(32):
            features.append(standard_features[0,:,:,i])
            features.append(blur_features[0,:,:,i])
            features.append(gray_features[0,:,:,i])
        names = ['standard', 'blur', 'gray']
        fig, ax = plt.subplots(8, 12, constrained_layout=True)
        for i in range(8):
            for j in range(12):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                ax[i,j].imshow(features[12*i+j], cmap='gray')
                ax[i,j].title.set_text(f"kernel {((12*i+j)//3)+1}: {names[(12*i+j)%3]}")
        fig.show()

    def first_pass(self):
        """
        Passes a dummy input through each model to prepare them for fast predictions.
        """
        first_input = np.zeros((1, 299, 299, 3))
        self._model.predict([first_input, first_input, first_input])

    def initialise_labels(self):
        """
        Creates a labels array, with each label's index corresponding to the model's prediction outputs.
        """
        self._labels = ["Amara lunicollis (id: 1035167)",     
                        "Amara aenea (id: 1035185)",
                        "Amara aulica (id: 1035194)",
                        "Amara familiaris (id: 1035195)",      
                        "Amara anthobia (id: 1035204)",        
                        "Amara bifrons (id: 1035208)",
                        "Amara apricaria (id: 1035231)",       
                        "Loricera pilicornis (id: 1035290)",   
                        "Patrobus septentrionis (id: 1035366)",
                        "Bradycellus harpalinus (id: 1035434)",
                        "Elaphrus lapponicus (id: 1035542)",
                        "Elaphrus riparius (id: 1035551)",
                        "Nebria brevicollis (id: 1035578)",
                        "Harpalus serripes (id: 1035864)",
                        "Bembidion testaceum (id: 1035929)",
                        "Bembidion tetracolum (id: 1035931)",
                        "Bembidion semipunctatum (id: 1036066)",
                        "Bembidion quadrimaculatum (id: 1036128)",
                        "Bembidion guttula (id: 1036154)",
                        "Bembidion obtusum (id: 1036192)",
                        "Bembidion properans (id: 1036203)",
                        "Bembidion lampros (id: 1036216)",
                        "Bembidion bruxellense (id: 1036255)",
                        "Sericoda quadripunctata (id: 1036286)",
                        "Carabus granulatus (id: 1036789)",
                        "Asaphidion flavipes (id: 1036796)",
                        "Clivina fossor (id: 1036893)",
                        "Clivina collaris (id: 1036899)",
                        "Anisodactylus binotatus (id: 1036917)",
                        "Miscodera arctica (id: 1036958)",
                        "Trechus rubens (id: 1037293)",
                        "Trechus obtusus (id: 1037319)",
                        "Acupalpus meridianus (id: 1037633)",
                        "Tachys scutellaris (id: 4308786)",
                        "Nebria complanata (id: 4308787)",
                        "Dicheirotrichus gustavii (id: 4308789)",
                        "Dicheirotrichus obsoletus (id: 4308790)",
                        "Bembidion fumigatum (id: 4308800)",
                        "Bembidion obscurellum (id: 4308801)",
                        "Bembidion aeneum (id: 4308804)",
                        "Bembidion iricolor (id: 4308805)",
                        "Bembidion minimum (id: 4308806)",
                        "Bembidion ephippium (id: 4308807)",
                        "Aepus marinus (id: 4308811)",
                        "Anisodactylus poeciloides (id: 4308812)",
                        "Broscus cephalotes (id: 4308815)",
                        "Carabus clatratus (id: 4470539)",
                        "Carabus problematicus (id: 4470555)",
                        "Carabus violaceus (id: 4470765)",
                        "Carabus monilis (id: 4470801)",
                        "Carabus nitens (id: 4471071)",
                        "Carabus glabratus (id: 4471113)",
                        "Masoreus wetterhallii (id: 4471202)",
                        "Panagaeus bipustulatus (id: 4471235)",
                        "Panagaeus cruxmajor (id: 4471238)",
                        "Pedius longicollis (id: 4471269)",
                        "Amara plebeja (id: 4472828)",
                        "Amara communis (id: 4472849)",
                        "Amara eurynota (id: 4472858)",
                        "Amara lucida (id: 4472884)",
                        "Amara convexior (id: 4472897)",
                        "Amara famelica (id: 4472900)",
                        "Amara spreta (id: 4472907)",
                        "Amara similata (id: 4472913)",
                        "Amara infima (id: 4472929)",
                        "Amara consularis (id: 4473127)",
                        "Zabrus tenebrioides (id: 4473277)",
                        "Stomis pumicatus (id: 4473297)",
                        "Elaphrus cupreus (id: 4473319)",
                        "Elaphrus uliginosus (id: 4473320)",
                        "Nebria rufescens (id: 4473617)",
                        "Nebria salina (id: 4473795)",
                        "Leistus fulvibarbis (id: 4473834)",
                        "Leistus ferrugineus (id: 4473860)",
                        "Leistus terminatus (id: 4473874)",
                        "Leistus spinibarbis (id: 4473903)",
                        "Olisthopus rotundatus (id: 4473974)",
                        "Oxypselaphus obscurus (id: 4474146)",
                        "Paranchus albipes (id: 4474169)",
                        "Synuchus vivalis (id: 4474653)",
                        "Drypta dentata (id: 4474780)",
                        "Brachinus crepitans (id: 4474861)",
                        "Odacantha melanura (id: 4474974)",
                        "Anisodactylus nemorivagus (id: 4474998)",
                        "Ophonus ardosiacus (id: 4475099)",
                        "Ophonus melletii (id: 4475127)",
                        "Ophonus schaubergerianus (id: 4475128)",
                        "Ophonus rufibarbis (id: 4475140)",
                        "Ophonus laticollis (id: 4475157)",
                        "Ophonus puncticeps (id: 4475179)",
                        "Ophonus rupicola (id: 4475181)",
                        "Ophonus cordatus (id: 4475188)",
                        "Ophonus azureus (id: 4475213)",
                        "Acupalpus dubius (id: 4475677)",
                        "Acupalpus brunnipes (id: 4475685)",
                        "Acupalpus exiguus (id: 4475707)",
                        "Dicheirotrichus cognatus (id: 4475757)",
                        "Dicheirotrichus placidus (id: 4475761)",
                        "Anthracus consputus (id: 4475794)",
                        "Licinus punctatulus (id: 4475846)",
                        "Patrobus atrorufus (id: 4475902)",
                        "Patrobus assimilis (id: 4475908)",
                        "Trechus quadristriatus (id: 4476667)",
                        "Trechoblemus micros (id: 4476832)",
                        "Blemus discus (id: 4478057)",
                        "Asaphidion stierlini (id: 4478679)",
                        "Asaphidion pallipes (id: 4478695)",
                        "Asaphidion curtum (id: 4478699)",
                        "Bembidion quadripustulatum (id: 4478842)",
                        "Bembidion illigeri (id: 4479052)",
                        "Syntomus truncatellus (id: 4479870)",
                        "Syntomus foveatus (id: 4479890)",
                        "Syntomus obscuroguttatus (id: 4479895)",
                        "Calodromius spilotus (id: 4480200)",
                        "Paradromius longiceps (id: 4480271)",
                        "Paradromius linearis (id: 4480302)",
                        "Philorhizus melanocephalus (id: 4480315)",
                        "Philorhizus notatus (id: 4480348)",
                        "Cicindela sylvatica (id: 4480480)",
                        "Cicindela campestris (id: 4480485)",
                        "Cicindela hybrida (id: 4480502)",
                        "Blethisa multipunctata (id: 4988207)",
                        "Calathus melanocephalus (id: 4988354)",
                        "Calathus fuscipes (id: 4988363)",
                        "Agonum thoreyi (id: 4988447)",
                        "Agonum muelleri (id: 4988484)",
                        "Pelophila borealis (id: 4988516)",
                        "Notiophilus substriatus (id: 5023047)",
                        "Notiophilus rufipes (id: 5023058)",
                        "Dyschirius salinus (id: 5716406)",
                        "Dyschirius obscurus (id: 5716408)",
                        "Dyschirius impunctipennis (id: 5716409)",
                        "Pogonus littoralis (id: 5716411)",
                        "Cychrus caraboides (id: 5753612)",
                        "Oodes helopioides (id: 5753653)",
                        "Poecilus cupreus (id: 5753697)",
                        "Poecilus versicolor (id: 5753720)",
                        "Poecilus lepidus (id: 5753725)",
                        "Poecilus kugelanni (id: 5753757)",
                        "Anchomenus dorsalis (id: 5754974)",
                        "Agonum marginatum (id: 5755011)",
                        "Agonum nigrum (id: 5755027)",
                        "Agonum ericeti (id: 5755044)",
                        "Agonum muelleri (id: 5755051)",
                        "Agonum emarginatum (id: 5755060)",
                        "Agonum versutum (id: 5755066)",
                        "Agonum viduum (id: 5755075)",
                        "Agonum fuliginosum (id: 5755079)",
                        "Agonum gracile (id: 5755080)",
                        "Agonum micans (id: 5755082)",
                        "Laemostenus terricola (id: 5755175)",
                        "Laemostenus complanatus (id: 5755203)",
                        "Calathus ambiguus (id: 5755293)",
                        "Calathus erratus (id: 5755296)",
                        "Calathus micropterus (id: 5755302)",
                        "Calathus cinctus (id: 5755322)",
                        "Harpalus latus (id: 5755339)",
                        "Calathus rotundicollis (id: 5755451)",
                        "Platyderus depressus (id: 5755456)",
                        "Polistichus connexus (id: 5755562)",
                        "Stenolophus teutonus (id: 5755950)",
                        "Stenolophus mixtus (id: 5755952)",
                        "Stenolophus skrimshiranus (id: 5755954)",
                        "Bradycellus caucasicus (id: 5755969)",
                        "Bradycellus sharpi (id: 5755978)",
                        "Bradycellus verbasci (id: 5755982)",
                        "Bradycellus ruficollis (id: 5755986)",
                        "Badister dilatatus (id: 5756010)",
                        "Badister unipustulatus (id: 5756015)",
                        "Badister bullatus (id: 5756018)",
                        "Badister sodalis (id: 5756021)",
                        "Pogonus luridipennis (id: 5756368)",
                        "Pogonus chalceus (id: 5756374)",
                        "Ocys quinquestriatus (id: 5756814)",
                        "Dyschirius impunctipennis (id: 5756874)",
                        "Dyschirius tristis (id: 5756945)",
                        "Cymindis axillaris (id: 5757120)",
                        "Cymindis vaporariorum (id: 5757155)",
                        "Dromius angustus (id: 5757192)",
                        "Dromius quadrimaculatus (id: 5757205)",
                        "Dromius agilis (id: 5757207)",
                        "Dromius meridionalis (id: 5757211)",
                        "Demetrias imperialis (id: 5757304)",
                        "Demetrias monostigma (id: 5757308)",
                        "Demetrias atricapillus (id: 5757310)",
                        "Ocys harpaloides (id: 5872109)",
                        "Bembidion lunulatum (id: 5872111)",
                        "Bembidion punctulatum (id: 5872119)",
                        "Bembidion articulatum (id: 5872124)",
                        "Bembidion tibiale (id: 5872127)",
                        "Bembidion decorum (id: 5872129)",
                        "Bembidion litorale (id: 5872143)",
                        "Bembidion gilvipes (id: 5872147)",
                        "Bembidion assimile (id: 5872148)",
                        "Perileptus areolatus (id: 5872219)",
                        "Carabus arcensis (id: 5872829)",
                        "Calosoma inquisitor (id: 5872954)",
                        "Harpalus rufipalpis (id: 5873211)",
                        "Harpalus attenuatus (id: 5873214)",
                        "Harpalus dimidiatus (id: 5873215)",
                        "Harpalus rubripes (id: 5873218)",
                        "Harpalus anxius (id: 5873220)",
                        "Microlestes maurus (id: 5873271)",
                        "Chlaenius vestitus (id: 5873357)",
                        "Dyschirius globosus (id: 5873524)",
                        "Dyschirius salinus (id: 5873525)",
                        "Callistus lunatus (id: 6097854)",
                        "Lebia chlorocephala (id: 6097857)",
                        "Dyschirius nitidus (id: 6097861)",
                        "Dyschirius politus (id: 6097862)",
                        "Bembidion bipunctatum (id: 6097864)",
                        "Bembidion prasinum (id: 6097865)",
                        "Bembidion mannerheimii (id: 6097867)",
                        "Bembidion nigricorne (id: 6097868)",
                        "Bembidion dalmatinum (id: 6097869)",
                        "Bembidion fluviatile (id: 6097871)",
                        "Bembidion monticola (id: 6097872)",
                        "Bembidion deletum (id: 6097878)",
                        "Bembidion femoratum (id: 6097879)",
                        "Bembidion atrocaeruleum (id: 6097882)",
                        "Bembidion varium (id: 6097883)",
                        "Sinechostictus stomoides (id: 6097885)",
                        "Bembidion lunatum (id: 6097886)",
                        "Bembidion stephensii (id: 6097890)",
                        "Bembidion normannum (id: 6097892)",
                        "Notiophilus germinyi (id: 6097896)",
                        "Notiophilus quadripunctatus (id: 6097897)",
                        "Cylindera germanica (id: 6097900)",
                        "Chlaenius nigricornis (id: 6097906)",
                        "Harpalus pumilus (id: 6097912)",
                        "Harpalus smaragdinus (id: 6097923)",
                        "Harpalus tardus (id: 6097924)",
                        "Harpalus laevipes (id: 6097926)",
                        "Harpalus neglectus (id: 6097927)",
                        "Epaphius secalis (id: 6097930)",
                        "Bembidion geniculatum (id: 7387665)",
                        "Pterostichus macer (id: 7416497)",
                        "Harpalus tenebrosus (id: 7419401)",
                        "Pterostichus strenuus (id: 7508714)",
                        "Elaphropus convexus (id: 7582513)",
                        "Pterostichus aethiops (id: 7613168)",
                        "Bembidion dentellum (id: 7639821)",
                        "Pterostichus madidus (id: 7641814)",
                        "Pterostichus oblongopunctatus (id: 7642452)",
                        "Tachys micros (id: 7683246)",
                        "Pterostichus nigrita (id: 7708350)",
                        "Cicindela maritima (id: 7727698)",
                        "Bembidion nigropiceum (id: 7792354)",
                        "Pterostichus minor (id: 7840196)",
                        "Pterostichus melanarius (id: 7856746)",
                        "Pterostichus diligens (id: 7873810)",
                        "Aepopsis robinii (id: 7888598)",
                        "Pterostichus cristatus (id: 7917481)",
                        "Pterostichus vernalis (id: 7975487)",
                        "Pterostichus gracilis (id: 8019310)",
                        "Bembidion clarkei (id: 8047965)",
                        "Nebria livida (id: 8051076)",
                        "Carabus nemoralis (id: 8056040)",
                        "Harpalus servus (id: 8068514)",
                        "Bembidion saxatile (id: 8104778)",
                        "Platynus assimilis (id: 8137203)",
                        "Elaphropus walkerianus (id: 8139233)",
                        "Pterostichus adstrictus (id: 8186956)",
                        "Trechus fulvus (id: 8236590)",
                        "Notiophilus palustris (id: 8264200)",
                        "Acupalpus parvulus (id: 8267077)",
                        "Bembidion schueppelii (id: 8300237)",
                        "Pterostichus quadrifoveolatus (id: 8307815)",
                        "Bembidion bualei (id: 8335452)",
                        "Harpalus serripes (id: 8378220)",
                        "Pterostichus rhaeticus (id: 8417764)",
                        "Pterostichus anthracinus (id: 8423772)",
                        "Amara ovata (id: 8563753)",
                        "Amara tibialis (id: 8600499)",
                        "Bembidion pallidipenne (id: 8607776)",
                        "Amara curta (id: 9027622)",
                        "Agonum piceum (id: 9206709)",
                        "Harpalus rufipes (id: 9251010)",
                        "Platynus livens (id: 9252314)",
                        "Amara convexiuscula (id: 9268484)",
                        "Pterostichus niger (id: 9314786)",
                        "Notiophilus aquaticus (id: 9346940)",
                        "Notiophilus biguttatus (id: 9364935)",
                        "Bembidion obliquum (id: 9377664)",
                        "Licinus depressus (id: 9414758)",
                        "Acupalpus flavicollis (id: 9415910)",
                        "Amara fulva (id: 9444427)",
                        "Bembidion laterale (id: 9479706)",
                        "Abax parallelepipedus (id: 9491931)",
                        "Bembidion doris (id: 9533851)",
                        "Calathus mollis (id: 9581584)"]

    def prep_image(self, image):
        """
        Augments and preprocesses an image before it is input to a model.
        """
        standard_img = preprocess_input(image)
        blurred_img = preprocess_input(self._blur_aug(image=image)['image'])
        grayed_img = preprocess_input(self._gray_aug(image=image)['image'])
        return [np.expand_dims(standard_img, axis=0), np.expand_dims(grayed_img, axis=0), 
                np.expand_dims(blurred_img, axis=0)]

    def model_predict(self, image, model):
        """
        Uses the selected model to predict the top 5 classes for an image.
        """
        results = model.predict(image)[0]
        values, indices = tf.nn.top_k(results, k=5)
        labels = [(self._labels[indices[i]], values[i]) for i in range(len(indices))]
        return labels
                
    def classify(self, image_path):
        """
        Prepares an image from a selected filename and predicts its class with both models.
        """
        inputs = self.prep_image(img_to_array(load_img(image_path, target_size=(299, 299))))
        labels = self.model_predict(inputs, self._model)
        inception_labels = self.model_predict(inputs[0], self._inception_model)
        return labels, inception_labels
