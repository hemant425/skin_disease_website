from flask import Flask, request, Response
from flask.templating import render_template
from flask import request
from werkzeug.utils import secure_filename
from app import app
import torch
from PIL import Image
import torchvision.transforms as T
import os
from model_architecture import SkinDiseaseModel

def predict(model, img, tr, classes):
    img_tensor = tr(img)
    out = model(img_tensor.unsqueeze(0))
    pred, idx = torch.max(out, 1)
    return classes[idx]

def get_transforms():
    transform = []
    transform.append(T.Resize((512, 512)))
    transform.append(T.ToTensor())
    return T.Compose(transform)

@app.route('/', methods=['GET', 'POST'])
def home_page():
    res = None
    if request.method == 'POST':
        classes = [
    "Acne_Keloidalis_Nuchae", "Acne_Vulgaris", "Acrokeratosis_Verruciformis",
    "Actinic_solar_Damage(Actinic_Cheilitis)", "Actinic_solar_Damage(Actinic_Keratosis)",
    "Actinic_solar_Damage(Cutis_Rhomboidalis_Nuchae)", "Actinic_solar_Damage(Pigmentation)",
    "Actinic_solar_Damage(Solar_Elastosis)", "Actinic_solar_Damage(Solar_Purpura)",
    "Actinic_solar_Damage(Telangiectasia)", "Acute_Eczema", "Allergic_Contact_Dermatitis",
    "Alopecia_Areata", "Androgenetic_Alopecia", "Angioma", "Angular_Cheilitis", "Aphthous_Ulcer",
    "Apocrine_Hydrocystoma", "Arsenical_Keratosis", "Balanitis_Xerotica_Obliterans",
    "Basal_Cell_Carcinoma", "Beau's_Lines", "Becker's_Nevus", "Behcet's_Syndrome",
    "Benign_Keratosis", "Blue_Nevus", "Bowen's_Disease", "Bowenoid_Papulosis",
    "Cafe_Au_Lait_Macule", "Callus", "Candidiasis", "Cellulitis", "Chalazion",
    "Clubbing_of_Fingers", "Compound_Nevus", "Congenital_Nevus", "Crowe's_Sign",
    "Cutanea_Larva_Migrans", "Cutaneous_Horn", "Cutaneous_T-Cell_Lymphoma",
    "Cutis_Marmorata", "Darier-White_Disease", "Dermatofibroma", "Dermatosis_Papulosa_Nigra",
    "Desquamation", "Digital_Fibroma", "Dilated_Pore_of_Winer", "Discoid_Lupus_Erythematosus",
    "Disseminated_Actinic_Porokeratosis", "Drug_Eruption", "Dry_Skin_Eczema",
    "Dyshidrosiform_Eczema", "Dysplastic_Nevus", "Eccrine_Poroma", "Eczema", "Epidermal_Nevus",
    "Epidermoid_Cyst", "Epithelioma_Adenoides_Cysticum", "Erythema_Ab_Igne",
    "Erythema_Annulare_Centrifigum", "Erythema_Craquele", "Erythema_Multiforme",
    "Exfoliative_Erythroderma", "Factitial_Dermatitis", "Favre_Racouchot", "Fibroma",
    "Fibroma_Molle", "Fixed_Drug_Eruption", "Follicular_Mucinosis", "Follicular_Retention_Cyst",
    "Fordyce_Spots", "Frictional_Lichenoid_Dermatitis", "Ganglion", "Geographic_Tongue",
    "Granulation_Tissue", "Granuloma_Annulare", "Green_Nail", "Guttate_Psoriasis",
    "Hailey_Hailey_Disease", "Half_and_Half_Nail", "Halo_Nevus", "Herpes_Simplex_Virus",
    "Herpes_Zoster", "Hidradenitis_Suppurativa", "Histiocytosis_X",
    "Hyperkeratosis_Palmaris_Et_Plantaris", "Hypertrichosis", "Ichthyosis", "Impetigo",
    "Infantile_Atopic_Dermatitis", "Inverse_Psoriasis", "Junction_Nevus", "Keloid",
    "Keratoacanthoma", "Keratolysis_Exfoliativa_of_Wende", "Keratosis_Pilaris", "Kerion",
    "Koilonychia", "Kyrle's_Disease", "Leiomyoma", "Lentigo_Maligna_Melanoma",
    "Leukocytoclastic_Vasculitis", "Leukonychia", "Lichen_Planus",
    "Lichen_Sclerosis_Et_Atrophicus", "Lichen_Simplex_Chronicus", "Lichen_Spinulosis",
    "Linear_Epidermal_Nevus", "Lipoma", "Livedo_Reticularis", "Lymphangioma_Circumscriptum",
    "Lymphocytic_Infiltrate_of_Jessner", "Lymphomatoid_Papulosis", "Mal_Perforans",
    "Malignant_Melanoma", "Median_Nail_Dystrophy", "Melasma", "Metastatic_Carcinoma", "Milia",
    "Molluscum_Contagiosum", "Morphea", "Mucha_Habermann_Disease", "Mucous_Membrane_Psoriasis",
    "Myxoid_Cyst", "Nail_Dystrophy", "Nail_Nevus", "Nail_Psoriasis", "Nail_Ridging",
    "Neurodermatitis", "Neurofibroma", "Neurotic_Excoriations", "Nevus_Comedonicus",
    "Nevus_Incipiens", "Nevus_Sebaceous_of_Jadassohn", "Nevus_Spilus", "Nummular_Eczema",
    "Onychogryphosis", "Onycholysis", "Onychomycosis", "Onychoschizia", "Paronychia",
    "Pearl_Penile_Papules", "Perioral_Dermatitis", "Pincer_Nail_Syndrome", "Pitted_Keratolysis",
    "Pityriasis_Alba", "Pityriasis_Rosea", "Pityrosporum_Folliculitis",
    "Poikiloderma_Atrophicans_Vasculare", "Pomade_Acne", "Pseudofolliculitis_Barbae",
    "Pseudorhinophyma", "Psoriasis", "Pustular_Psoriasis", "Pyoderma_Gangrenosum",
    "Pyogenic_Granuloma", "Racquet_Nail", "Radiodermatitis", "Rhinophyma", "Rosacea",
    "Scalp_Psoriasis", "Scar", "Scarring_Alopecia", "Schamberg's_Disease",
    "Sebaceous_Gland_Hyperplasia", "Seborrheic_Dermatitis", "Seborrheic_Keratosis",
    "Skin_Tag", "Solar_Lentigo", "Stasis_Dermatitis", "Stasis_Edema", "Stasis_Ulcer",
    "Steroid_Acne", "Steroid_Striae", "Steroid_Use_abusemisuse_Dermatitis", "Stomatitis",
    "Strawberry_Hemangioma", "Striae", "Subungual_Hematoma", "Syringoma", "Terry's_Nails",
    "Tinea_Corporis", "Tinea_Cruris", "Tinea_Faciale", "Tinea_Manus", "Tinea_Pedis",
    "Tinea_Versicolor", "Toe_Deformity", "Trichilemmal_Cyst", "Trichofolliculoma",
    "Trichostasis_Spinulosa", "Ulcer", "Urticaria", "Varicella", "Verruca_Vulgaris",
    "Vitiligo", "Wound_Infection", "Xerosis"
]

        
        f = request.files['file']
        filename = secure_filename(f.filename)

        UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads')
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        f.save(path)  # ✅ SAVE THE FILE

        # Load model
        model = SkinDiseaseModel(num_classes=198)
        model.load_state_dict(torch.load(r"C:\Users\heman\OneDrive\Documents\Python projects\Mini project-5 sem\Jarvis\project trail 1\skinnnn\Skin-Disease-Detection-master\efficientnet_sd198_model.pt", map_location='cpu')) 
        weights_only=False  # make sure you're using this correctly if model is not weights-only
        model.eval()
        device = torch.device('cpu')
        model.to(device)

        # Read image
        img = Image.open(path).convert("RGB")
        tr = get_transforms()
        res = predict(model, img, tr, classes)

    return render_template("index.html", res=res)

