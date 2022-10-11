import spacy
from collections import Counter
from string import punctuation
from string import digits
import pandas as pd

nlp = spacy.load("es_dep_news_trf")

def get_hotwords(text):
    result = []
    pos_tag = ['NOUN','VERB'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation or token.text in digits):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result
text = """
1. Esta es una oportunidad de becas para todos los ciudadanos de los Estados Miembros de la OEA.,2. Esta oportunidad de beca está abierta para nuevos solicitantes, así como para estudiantes actuales de CSU que sean de un estado miembro de la OEA y no un beneficiario actual de la beca OEA-CSU.,3. Envíe una solicitud de beca completa con todos 
los documentos de respaldo requeridos,4. Ser ciudadano y / o residente legal permanente de cualquier estado miembro de la OEA, excepto Estados Unidos.,5. Proporcionar una carta de admisión incondicional de CSU.,LOS SIGUIENTES SOLICITANTES NO SON ELEGIBLES PARA ESTE PROGRAMA:,a. Becarios actuales de la OEA,b. Personal de la SG / OEA y sus familiares inmediatos o consultores de la SG / OEA,c. Funcionarios permanentes de la OEA y sus familiares inmediatos.
¿QUIÉNES PUEDEN POSTULARSE A ESTE PROGRAMA DE BECAS?,1. Requisito básico para realizar la estancia bajo la modalidad binacional: El postulante ya debe estar cursando un Doctorado o contar con la admisión a un Doctorado en una universidad del país de origen (por regla general una universidad colombiana, ecuatoriana, peruana o venezolana).,2. El director de tesis de la universidad de origen debe dar su carta de aval o aceptación para realizar la estancia en Alemania.,3.
Adicionalmente, el postulante debe contar con una aceptación/invitación de un supervisor de una universidad alemana.,4. Requisito básico para realizar la estancia bajo la modalidad Cotutela: Además de las condiciones indicadas para la modalidad binacional, es necesario que exista un convenio de cooperación suscrito entre la universidad de origen y la universidad alemana, en el que se establezcan todas las pautas para la supervisión y el otorgamiento de una doble titulación.,
5. También pueden participar candidatos que aún estén en el proceso de admisión al Doctorado en su país de origen. Si la beca es otorgada, a más tardar en el momento de su inicio, el candidato debe demostrar que ya cuenta con la carta de admisión definitiva al Doctorado en la universidad de origen y que el Doctorado se realizará bajo la modalidad de cotutela con la respectiva universidad alemana.,¿QUÉ TÍTULO ACADÉMICO SE DEBE TENER PARA POSTULARSE A ESTE PROGRAMA DE BECAS?,
a. Por regla general para realizar una estancia doctoral en Alemania es necesario contar con un título de Maestría, y el DAAD requiere que en la postulación se demuestre que ya se completó este nivel de estudios.,b. Las Especializaciones o Diplomados no reemplazan ese i.,c. Sin embargo, las personas que ya están cursando el Doctorado o que fueron admitidos para iniciar uno, aún sin tener título de Maestría, podrían presentarse a esta convocatoria.,¿EXISTE UN LÍMITE DE EDAD PARA POSTULARSE A ESTA CONVOCATORIA?,1. El DAAD no estipula un límite de edad para participar, pero es i que el último título universitario (normalmente Maestría) no tenga una antigüedad superior a 6 años en el momento de la fecha de cierre de esta convocatoria. Si el candidato ya está realizando un Doctorado, la fecha de inicio de éste no debe superar los 3 años al momento del cierre de esta convocatoria.,
2. Únicamente en casos especiales el DAAD tiene definidas algunas excepciones que permitirían postularse con un título más antiguo; por ejemplo, por motivos de fuerza mayor justificados por enfermedades crónicas o por el cuidado de hijos pequeños. Consulte la siguiente página para comprobar si una de las excepciones aplica en su caso: www.daad.de/en/study-and-research-in- germany/scholarships/important-information-for-scholarship-applicants/ (Sección A punto 2),3. Si su caso hace parte de una de estas excepciones y usted decide postularse, debe explicarlo en el último espacio del formulario de la postulación en el DAAD Portal. El DAAD puede solicitarle documentos de soporte para comprobar sus circunstancias particulares (ejemplo: historias clínicas que evidencien que tuvo o tiene una enfermedad, que sufrió un accidente, etc.).,4. ¿Se necesita tener la aceptación del director de tesis/supervisor de Alemania?,
5. En la postulación se deberán incluir las cartas de aceptación de los dos directores/supervisores de tesis; la del director de la universidad de origen y la del director/supervisor de la universidad alemana, quiénes supervisarán conjuntamente la investigación doctoral. Las características de esas cartas se explican en la presente guía, en la sección “Indicaciones sobre la documentación requerida”.,¿EN QUÉ IDIOMA SE PUEDE REALIZAR LA ESTANCIA DOCTORAL?,a. En Alemania normalmente se puede realizar estancias de investigación doctoral en alemán o en inglés. Cada universidad, centro de investigación o supervisor de tesis define el nivel de conocimientos de idioma y el tipo de certificado que solicita. El postulante es responsable de confirmar cuáles son esos requisitos directamente con la universidad, centro de investigación o director/supervisor de tesis con quien planea realizar la estancia en Alemania.,
b. Para presentarse a esta convocatoria el DAAD no exige un certificado específico de idiomas, pero la postulación debe incluir alguna certificación que evidencie el cumplimiento de las siguientes condiciones:,c. Para realizar estancias en inglés: Contar con un nivel de comunicación fluido de manera oral y escrito. Para realizar investigaciones doctorales en inglés, el DAAD no exige conocimientos de alemán.,d. Para realizar estancias en alemán: El DAAD no exige un examen o puntaje específico, pero en este caso el postulante debe contar con un nivel de comunicación fluido de alemán de manera oral y escrito. Normalmente, los candidatos de algunas áreas del conocimiento como Derecho o Ciencias Humanas y Sociales, deben contar con muy buenos conocimientos de alemán desde el inicio de su postulación. Pero esto lo debe consultar el candidato directamente con la universidad alemana anfitriona o el director/supervisor de tesis de Alemania.,
e. Para realizar estancias en español: Solo en casos excepcionales y en ciertas áreas del conocimiento (por ejemplo: Estudios Latinoamericanos o hispanoamericanos, Romanística), se pueden llevar a cabo investigaciones doctorales en español. Sin embargo, en estos casos el DAAD solicita que el candidato no solo pueda comunicarse en español, sino que además tenga conocimientos de inglés o de alemán. Adicionalmente, los candidatos que planean realizar su estancia de investigación en español deben informarse con la universidad alemana a la que está vinculado el director/supervisor de tesis, sobre las condiciones oficiales del idioma requerido para la escritura y presentación del documento oficial producto de la estancia de investigación.,f. Si la investigación doctoral en Alemania se realizará completamente en inglés, ¿se debe demostrar conocimientos de alemán?,g. Si la estancia de investigación en Alemania se llevará a cabo completamente en inglés, no se necesita demostrar conocimientos de alemán ante el DAAD.,¿ES POSIBLE POSTULARSE DESDE ALEMANIA?,1. Sí es posible, siempre y cuando el candidato no lleve viviendo en Alemania más de 15 meses en el momento de la fecha de cierre de la convocatoria. Si resulta seleccionado, el DAAD no le reembolsará el valor de ningún gasto en el que haya incurrido antes de la confirmación oficial de la beca.,2. Esta restricción no aplica si el candidato vivió previamente en Alemania y ya está radicado nuevamente en su país de origen en el momento de la postulación.,¿ES POSIBLE POSTULARSE SI YA SE ESTÁ REALIZANDO UN DOCTORADO EN ALEMANIA?,
a. No. Este programa tiene previsto apoyar a los candidatos que estén matriculados en un Doctorado,b. en un país diferente a Alemania (por regla general una universidad colombiana, ecuatoriana, peruana o venezolana).,¿PUEDEN POSTULARSE CANDIDATOS QUE ESTÉN VIVIENDO EN OTRO PAÍS, QUE NO ES EL PAÍS DE ORIGEN NI ALEMANIA?,1. En caso tal que desde hace mínimo un año el candidato está radicado en un país diferente al de su origen (tampoco Alemania), y/o haya recibido en el país en el que actualmente vive su último título universitario, primero deberá contactar al DAAD de dicho país y confirmar si puede postularse a través de los programas de becas que estén disponibles para ese país o región. En esa comunicación deberá informar el tiempo de residencia y si está estudiando o ya obtuvo un título universitario allí.,2. Si le informan que una postulación no es posible, debe reenviar esa respuesta escrita a la Oficina Regional del DAAD en Bogotá para evaluar la viabilidad de su participación en esta convocatoria.,¿ES POSIBLE POSTULARSE A ESTA BECA TENIENDO ALGUNA DISCAPACIDAD O ENFERMEDAD CRÓNICA?,3. El DAAD apoya el llamado de la Convención de las Naciones Unidas sobre los Derechos de las Personas con Discapacidad (2009). Por eso, el DAAD invita a las personas con discapacidades o enfermedades crónicas a presentar su postulación. En caso de obtener la beca, el DAAD podría asumir una determinada cantidad de los gastos adicionales ocasionados en Alemania por una discapacidad o enfermedad crónica, que no puedan ser cubiertos por ningún otro portador de gastos; por ejemplo, el seguro médico. Si tiene más preguntas sobre el tema de la inclusión y la igualdad de oportunidades, puede escribir en inglés sobre su caso concreto a: diversity@daad.de.,INFORMACIÓN PARA CANDIDATOS DE MEDICINA, VETERINARIA Y ODONTOLOGÍA,a. Si es egresado de Medicina, Medicina Veterinaria u Odontología, debe tener en cuenta que el DAAD solo financia Doctorados en áreas que no correspondan al ámbito clínico o quirúrgico y orientar su postulación de acuerdo con la información disponible en los siguientes enlaces:,b. Inglés: Additional Information on DAAD Research Grants for Applicants from Medical Fields,c. Alemán: Zusätzliche Hinweise für DAAD-Forschungsstipendien für Bewerber aus medizinischen Fachbereichen
"""

#print(output)

df = pd.read_csv('Datasets_procesados/DT_becas_noNans_noIndex_copy.csv')
requisitos = df['requirements']

"""for i in range(3):
    print(type(requisitos[i]))"""

    
#output = set(get_hotwords(text))
#output= set()
keywords = []
for i in range(10):
    #output.update(get_hotwords(requisitos[i]))
    output = set(get_hotwords(requisitos[i]))
    most_common_list = Counter(output).most_common(15)
    keywords.append(list(map(lambda x: x[0],most_common_list)))


df['keywords'] = keywords
print(df['keywords'])

#TO DO APLICAR KEYWORDS A TODAS LAS FILAS DEL DT
#HACERLOS SOBRE UNA COPIA DEL DT
# GUARDAR EL NUEVO DT Y APLICAR TF-IDF

#print(output)
#most_common_list = Counter(output).most_common(30)

"""for item in most_common_list:
  print(item[0])"""