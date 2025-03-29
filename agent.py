# Copyright (c) Microsoft. All rights reserved.

import asyncio

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

"""
The following sample demonstrates how to use an already existing
Azure AI Agent within Semantic Kernel. This sample requires that you
have an existing agent created either previously in code or via the
Azure Portal (or CLI).
"""

# Simulate a conversation with the agent
USER_INPUTS = [
text = """
    Motif d'admission 23/10
    rééducation et réautonomisation chez une patiente de 83 ans dans les suites d'une osteosynthèse sur fracture periprothétique du femur proximal gauche avec mise en place d'un cerclage sur PTH gauche le et prise en charge chirurgicale d'un HSD chronique

    Antécédents médicaux
    HTA
    trouble du sommeil

    Antécédents chirurgicaux
    PTH G il y a 10 ans sur chute

    Antécédents transfusionnels
    NC
    Maladies en cours
    fracture periprothétique du femur gauche
    hematome sous dural chronique drainé
    trouble de la marche
    perte d'autonomie
    traitement anticoagulant préventif

    Allergies alimentaires
    averte viande

    Histoire de la maladie
    Patiente adressée aux urgences pour chute au domicile avec amnésie des faits.
    transfert le 7/10/23 devant des troubles de la conscience et un deficit hemicorporel gauche en rapport avec un hematome sous dural droit avec effet de masse visualisé sur un scanner.
    Mise en évidence également d'une fracture du col gauche peri prothétique.

    -Intervention neurochirurgicale réalisée le 09/10/23 : évacuation de l'hematome par
    Pas de complication dans les suites.
    Traitement par KEPPRA 250mg matin et soir pendant 3 semaines.
    RDV de controle prévu le 06/12/23 avec TDM crane sans injection.

    -Ostéosynthèse du femur proximal gauche : mise en place de fils de cerclage metallique le 17/10/23.
    Ablation des agraphes dans 20j / soisn locaux toutes les 48h.
    Consultation de suivi ortho le 5/10/23 avec le Dr avec radio de contrôle.
    Appui total autorisé. Verticalisation faite avec kiné à

    Traitement personnel à l'entrée
    VALSARTAN/HYDOCHLOROTHIAZIDE 160/12.5mg 1 par jour
    BISOCE 5mg par jour

    XARELTO 10mg par jour
    DOLIPRANE 500mg
    PANTOPRAZOLE 40mg
    HAVLANE 1la nuit

    Mode de vie
    vit seule au domicile.
    Autonomie préservée pour les AVQ.
    Conduisait sa voiture avant la chute.
    pas d'aide au domicile.
    1 fille présente qui vit à Marseille, 2 fils à Montreal.

    Orientation souhaitée à la sortie
    RAD selon évolution

    Examen d'entrée

    Etat général : preservée
    Variation du poids : NC
    Urodigestif : abdo souple depressible indolore
    pas de selles depuis 10j seuement des traces, GAZ +
    Pulmonaire : plages pulmonaires libres
    Cardio-vasculaire : BDC réguliers, pas de souffle, mollets souples
    Psychiatrique : pas de dysthymie
    Neurologique : G15, pas de deficit moteur à droite
    paresthésie MIG (opéré), force motrice moins importante à gauche.
    Pas de paralysie faciale.
    Locomotion : marche non évaluée
    Cutanée : ras
    Vision : RAS
    Etat buccodentaire : pas de mycose
    prothèse dentaire mais pas de dentier
    Audition : RAS

    Examen d'entrée EGT V3

    EVS : 0
    ALGOPLUS :
    CONCLUSION DE L'EXAMEN CLINIQUE :

    Patiente de 83 ans en rééducattion post opératoire d'un HSD chronique operé et d'une fracture periprothetique du col femoral gauche

    Patient à risque de troubles de la déglutition : non
    PROJET THERAPEUTIQUE INITIAL : Prise en charge medicale :
    Bilan de chute : holter ECG, echo doppler TSAO, bilan cardiologique

    Surveillance biologiques et des traitements

    Prise en charge kiné :
    réautonomisation à la marche

    Rééducation et réadapation dans les suites d'une osteosynthèse du femur gauche par plaque le 21/08/23 par le Dr chez une patiente de 87 ans.

    Antécédents médicaux
    fibrillation atriale emboligène (AVC ischémique)
    cardiopathie hypokinétique
    HTA
    VPPB
    fracture du radius droit

    Antécédents chirurgicaux
    prothèse intermédiaire de hanche droite
    cholecystectomie
    appendicectomie
    fracture femur gauche osteosynthésée

    Antécédents transfusionnels
    1 CGR aout 2023

    Maladies en cours
    facture femur gauche
    syndrome anxieux

    Allergies médicamenteuses
    0

    Allergies alimentaires
    0

    Histoire de la maladie
    Chute traumatique au domicile le 14/08/23.
    --> fracture du femur gauche sous une PIH
    --> osteosynthèse par plaque le 21/08/23 : suites simples
    Consignes de sortie :
    -appui non autorisé / immobilisation par ZIMMER 3 mois au total
    -Prevention TVP par HBPM 45j
    -Refection pansement tous les 2 jours
    -ablation des agraphes dans 21j à partir du 4 septembre 2023.

    RDV le 3 octobre 2023 avec le Dr LIMOUZIN à 10h avec radio de contrôle

    Traitement personnel à l'entrée
    KARDEGIC Acétylsalicylate sachet 75 mg - Voie : PO - Posologie : 75mg par 24 heures-
    DOLIPRANE Paracétamol gel 500mg - Voie : PO- Posologie : 1g par 8 heures-
    PANTOPRAZOLE (Cp 20mg) - Voie: PO- Posologie: 20mg à 19:00
    TRIATEC Ramipril cp 5mg - Voie: PO - Posologie : 10mg à 8:00
    MIANSERINE cp 30mg- Voie: PO - Posologie : 20mg à 20:00
    ESIDREX Hydrochlorothiazide cp 25mg - Voie : PO- Posologie: 25mg à 8:00
    intro izalgi matin et soir
    intro macrogol car me dit ne pas avoir de selles depuis 2 jours

    28/11/2023
    HTA malgré bithérapie optimale, demande d'une CS cardio, courrier fait.

    29/11/2023
    : absente en chmabre

    30/11/2023
    Staff pluriprofessionnel
    IDE du service
    ASDE du service
    Kinésithérapeute
    Dépendance complète
    Douleur jambe gauche non calmée par antalgiques de palier 2 -> passage au palier 3 et à réévaluer
    A marché ce matin la longueur de barre mais s'arrête car douloureuse
    Pleurs à la mobilisation de son genou gauche
    Doute sur algodystrophie du genou gauche, consultation rhumato à prévoir, raideur apparue 3 jours après retrait attelle malgré mobilisation en actif et passif dès le début par kiné ? courrier fait
    Diffiuclté à la prise en charge par les équipes, n’a pas la volonté et se plaint régulièrement
    Orienter vers EHPAD ++, famille envisageait RAD mais à revoir car impossible selon les équipes et ne pourra à priori pas remarcher.

    06/12/2023

    Bon contact, calme, anxiété palpable.
    Discours clair, cohérent et adapté, libre de tout délire, sans désorganisation.
    Thymie basse, idées noires, idées suicidaires non scénarisées, sans velléité de passage à l'acte. Absence de projection dans le service.
    Sommeil amélioré sous ttt
    Appétit conservé.
    Trouble de la personnalité dépressive depuis qu'elle est jeune.
    CAT : proposition d'augmenter MIRTAZAPINE à 30 mg, surveillance risque de chute, HTO.

    Visite hebdomadaire
    reeducation post osteosynthese sur plaque femur gauche chez patiente au profil chuteur multiple ( FA, VPPNB HTA)

    sur le plan general et fonctionnel
    oms 3
    dependante pr AVQ
    alterne entre lit et fauteuil

    sur le plan orthopedique

    plaie propre
    algique en regard plaque
    reprise appui depuis 3semaines

    sur le plan thymique
    tristesse humeur sur douleur
    =>majoration mirtazapine 3cp

    sur le plan social
    orientation a definir en fonction reautonomisation ehpad vs RAD
    13/12/2023
    : suivi
    légère amélioration sur le plan thymique, moins d'angoisse, persistance d'idées noires sur personnalité dépressive. Pas d'idée suicidaire.
    Fonctions instinctuelles préservées.
    Trouve la rééducation longue.
    CAT : poursuite ttt, pas de modif, + suivi


  Visite hebdomadaire // Cardiologie 
      avis cardio indicattion theorique reprise naco devant avc ischemique sur FA 
devant risque chutes la balance benefices risques est en defaveur de la reprise
 
18/12/2023 
    controle osteosynthese
consolidation ok 
appui autorise mais se deplace peu 
poursuite stimulation verticalisation 

projet reeducation a reevaluer cette semaine avec kine et equipe soignante, patiente tres difficile a prendre en charge , refuse tout actiivite.
 

- FIN -  


Dépistage des patients à risque BHRe
Patient hospitalisé pendant au moins 24h ou pris en charge dans une filière spécifique (dialyse ou autre) à l’étranger durant les 12 derniers mois ? : Non
    """
    ,
]

async def main() -> None:
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        # 1. Retrieve the agent definition based on the `agent_id`
        # Replace the "your-agent-id" with the actual agent ID
        # you want to use.
        agent_definition = await client.agents.get_agent(
            agent_id="asst_E5nFroutEcRYyKkXsLkMwPvJ",
        )

        # 2. Create a Semantic Kernel agent for the Azure AI agent
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
        )

        # 3. Create a thread for the agent
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread: AzureAIAgentThread = None

        try:
            for user_input in USER_INPUTS:
                print(f"# User: '{user_input}'")
                # 4. Invoke the agent for the specified thread for response
                response = await agent.get_response(messages=user_input, thread=thread)
                print(f"# {response.name}: {response}")
        finally:
            # 5. Cleanup: Delete the thread and agent
            await thread.delete() if thread else None
            # Do not clean up the agent so it can be used again


if __name__ == "__main__":
    asyncio.run(main())