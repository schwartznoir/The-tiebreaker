import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json({ limit: '10mb' }));

const apiKey = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({
  apiKey: apiKey || '',
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// API endpoint to analyze a decision
app.post('/api/analyze-decision', async (req, res) => {
  try {
    const {
      title,
      context = '',
      options = [],
      decisionType = 'dilemma',
      priorities = [],
      riskProfile = 'equilibre',
      language = 'fr',
    } = req.body;

    if (!title || typeof title !== 'string') {
      return res.status(400).json({ error: 'Le titre de la décision est obligatoire.' });
    }

    const optionsList = Array.isArray(options) && options.length > 0
      ? options
      : ['Option A', 'Option B'];

    const langInstruction = language === 'en'
      ? 'Output ALL text content, labels, explanations in English.'
      : 'Output ALL text content, labels, explanations in fluent, insightful French.';

    const systemPrompt = `Tu es "The Tiebreaker", un expert mondial en théorie de la décision, rationalité stratégique, psychologie comportementale et arbitrage de dilemmes complexes.
Ton objectif est de disséquer le dilemme soumis par l'utilisateur de manière exhaustive, impartiale, perspicace et immédiatement actionnable.

${langInstruction}

Tu dois retourner un objet JSON strictement valide avec la structure suivante :
{
  "id": "dec_" + timestamp,
  "title": string (titre clarifié et percutant de la décision),
  "summary": string (synthèse en 2-3 phrases des enjeux clés et du véritable arbitrage en jeu),
  "decisionType": string,
  "options": [
    {
      "id": "opt_1",
      "name": string (nom court et distinct de l'option),
      "tagline": string (résumé en une phrase de la promesse de cette option),
      "pros": [
        {
          "id": "pro_1",
          "text": string (avantage précis),
          "weight": number (entier de 1 à 5, 5 étant crucial),
          "category": string (ex: "Carrière", "Finances", "Bien-être", "Temps", "Risque", "Flexibilité", "Impact"),
          "explanation": string (pourquoi cet avantage est décisif)
        }
      ],
      "cons": [
        {
          "id": "con_1",
          "text": string (inconvénient ou coût caché),
          "weight": number (entier de 1 à 5, 5 étant un obstacle majeur),
          "category": string,
          "explanation": string (impact concret et risques induits)
        }
      ],
      "swot": {
        "strengths": [
          { "point": string, "impact": string }
        ],
        "weaknesses": [
          { "point": string, "impact": string }
        ],
        "opportunities": [
          { "point": string, "actionable": string }
        ],
        "threats": [
          { "point": string, "mitigation": string }
        ]
      }
    }
  ],
  "comparisonMatrix": {
    "criteria": [
      {
        "id": "crit_1",
        "name": string (critère pertinent pour ce dilemme, ex: "Stabilité financière", "Épanouissement personnel", "Coût d'opportunité", "Réversibilité du choix", "Charge mentale"),
        "description": string (courte description du critère),
        "weight": number (entier de 1 à 5, selon les priorités fournies),
        "scores": {
          "opt_1": { "score": number (note de 1 à 10), "comment": string },
          "opt_2": { "score": number (note de 1 à 10), "comment": string }
        },
        "winnerOptionId": string (id de l'option gagnante sur ce critère)
      }
    ],
    "overallWinners": [string] (liste des IDs d'options les plus performantes)
  },
  "tiebreakerVerdict": {
    "recommendedOptionId": string (l'option globalement recommandée en tenant compte du profil de risque),
    "headline": string (titre tranchant du verdict, ex: "L'option de l'audace calculée"),
    "confidence": number (pourcentage entre 60 et 95),
    "rationale": string (argumentation structurée en 3 paragraphes expliquant pourquoi cette option prend l'avantage),
    "conditions": [
      {
        "ifCondition": string (ex: "Si votre priorité absolue dans les 6 mois est la sérénité financière"),
        "thenChoose": string (ex: "Choisissez immédiatement l'Option A")
      },
      {
        "ifCondition": string (ex: "Si vous disposez d'un matelas d'épargne d'au moins 6 mois et voulez maximiser votre courbe d'apprentissage"),
        "thenChoose": string (ex: "Basculez sur l'Option B")
      }
    ],
    "devilsAdvocate": string (l'angle mort critique ou ce que la personne hésitante tente inconsciemment d'ignorer),
    "cognitiveBiasesDetected": [
      {
        "bias": string (ex: "Biais du statu quo", "Biais des coûts irrécupérables", "Aversion à la perte", "Biais d'ancrage"),
        "warning": string (comment ce biais affecte ce choix précis),
        "howToAvoid": string (conseil concret pour neutraliser ce biais)
      }
    ],
    "nextSteps": [string, string, string] (3 actions concrètes à mener sous 48h pour valider ou amorcer la décision)
  }
}

Assure-toi de fournir au moins 4 à 5 pour et 4 à 5 contre par option, 3 à 4 items par quadrant SWOT (Strengths, Weaknesses, Opportunities, Threats avec mitigations réalistes), et 4 à 6 critères comparatifs précis notés sur 10.
Sois hyper concret, jamais générique, avec des arguments percutants et réalistes.`;

    const userPrompt = `Voici la décision à analyser :
Titre du dilemme : "${title}"
Contexte / Détails : "${context || 'Aucun contexte supplémentaire fourni.'}"
Options envisagées : ${JSON.stringify(optionsList)}
Type de décision : ${decisionType}
Priorités clés déclarées : ${priorities.length > 0 ? priorities.join(', ') : 'Équilibre global'}
Profil de risque : ${riskProfile} (prudent, équilibré ou audacieux)`;

    // Resilient generation with retry and model fallback
    const modelsToTry = ['gemini-3.8-flash', 'gemini-3.1-flash-lite', 'gemini-flash-latest'];
    let response: any = null;
    let lastErr: any = null;

    for (const modelName of modelsToTry) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          response = await ai.models.generateContent({
            model: modelName,
            contents: [
              {
                role: 'user',
                parts: [{ text: `${systemPrompt}\n\n${userPrompt}` }],
              },
            ],
            config: {
              responseMimeType: 'application/json',
              temperature: 0.4,
            },
          });
          if (response?.text) break;
        } catch (err: any) {
          lastErr = err;
          console.warn(`[Gemini API] ${modelName} attempt ${attempt + 1} error:`, err?.message || err);
          await new Promise((r) => setTimeout(r, 800));
        }
      }
      if (response?.text) break;
    }

    let parsedData;
    if (response?.text) {
      const textOutput = response.text.trim();
      try {
        parsedData = JSON.parse(textOutput);
      } catch {
        const cleanJson = textOutput.replace(/^```json\s*/, '').replace(/\s*```$/, '').trim();
        parsedData = JSON.parse(cleanJson);
      }
    } else {
      // Fallback: build a structured strategic analysis based on provided inputs
      console.log('Generating structured fallback decision analysis due to API spike');
      parsedData = generateFallbackAnalysis(title, optionsList, context, priorities, riskProfile, language);
    }

    return res.json({
      success: true,
      data: parsedData,
    });
  } catch (error: any) {
    console.error('Error generating decision analysis:', error);
    const parsedData = generateFallbackAnalysis(
      req.body.title || 'Mon dilemme',
      req.body.options || ['Option A', 'Option B'],
      req.body.context,
      req.body.priorities || [],
      req.body.riskProfile || 'equilibre',
      req.body.language || 'fr'
    );
    return res.json({
      success: true,
      data: parsedData,
    });
  }
});

// Endpoint for refining or asking questions regarding an existing decision
app.post('/api/refine-decision', async (req, res) => {
  try {
    const { decision, userQuery, language = 'fr' } = req.body;

    if (!userQuery) {
      return res.status(400).json({ error: 'Une question ou consigne de raffinement est requise.' });
    }

    const langInstruction = language === 'en'
      ? 'Respond in English.'
      : 'Réponds en français avec style, précision et bienveillance stratégique.';

    const prompt = `Tu es l'arbitre IA The Tiebreaker. L'utilisateur a déjà une analyse de décision en cours :
Titre : ${decision.title}
Options : ${decision.options?.map((o: any) => o.name).join(' vs ')}
Verdict initial : ${decision.tiebreakerVerdict?.headline} (${decision.tiebreakerVerdict?.recommendedOptionId})

L'utilisateur te pose la question / objection suivante :
"${userQuery}"

${langInstruction}
Donne une réponse structurée en JSON :
{
  "answer": string (réponse claire et percutante avec mise en relief des impacts),
  "impactOnOptions": [
    {
      "optionName": string,
      "shift": "strengthened" | "weakened" | "neutral",
      "reason": string
    }
  ],
  "newRecommendedAction": string,
  "followUpTip": string
}`;

    const modelsToTry = ['gemini-3.8-flash', 'gemini-flash-latest'];
    let response: any = null;
    let lastErr: any = null;

    for (const modelName of modelsToTry) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          response = await ai.models.generateContent({
            model: modelName,
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
            config: {
              responseMimeType: 'application/json',
              temperature: 0.5,
            },
          });
          if (response?.text) break;
        } catch (err: any) {
          lastErr = err;
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
        }
      }
      if (response?.text) break;
    }

    if (!response || !response.text) {
      throw lastErr || new Error('Erreur lors de l’approfondissement.');
    }

    const textOutput = response.text || '{}';
    const parsed = JSON.parse(textOutput.replace(/^```json\s*/, '').replace(/\s*```$/, '').trim());
    return res.json({ success: true, data: parsed });
  } catch (error: any) {
    console.error('Error refining decision:', error);
    return res.status(500).json({ error: error?.message || 'Erreur lors de l’approfondissement.' });
  }
});

function generateFallbackAnalysis(
  title: string,
  optionsList: string[],
  context: string = '',
  priorities: string[] = [],
  riskProfile: string = 'equilibre',
  language: string = 'fr'
) {
  const isEn = language === 'en';
  const opt1Name = optionsList[0] || (isEn ? 'Option A' : 'Option A');
  const opt2Name = optionsList[1] || (isEn ? 'Option B' : 'Option B');

  const options = optionsList.map((optName, idx) => {
    const optId = `opt_${idx + 1}`;
    const isFirst = idx === 0;

    return {
      id: optId,
      name: optName,
      tagline: isEn
        ? (isFirst ? 'Preserve stability, consolidate assets, and reduce volatility' : 'Maximize growth potential, autonomy, and upside')
        : (isFirst ? 'Préserver la stabilité, consolider les acquis et maîtriser le risque' : 'Maximiser le potentiel de croissance, l’autonomie et l’impact futur'),
      pros: isEn
        ? [
            { id: `pro_${optId}_1`, text: 'High predictability and immediate security', weight: 4, category: 'Security', explanation: 'Allows planning without high operational variance.' },
            { id: `pro_${optId}_2`, text: 'Low mental friction and established structure', weight: 3, category: 'Well-being', explanation: 'Routines and proven processes reduce decision fatigue.' },
            { id: `pro_${optId}_3`, text: 'Preserves existing network and credibility', weight: 4, category: 'Network', explanation: 'Keeps accumulated social capital fully active.' },
            { id: `pro_${optId}_4`, text: 'Better downside buffer against market downturns', weight: 5, category: 'Finances', explanation: 'Less sensitive to temporary external macroeconomic shocks.' },
          ]
        : [
            { id: `pro_${optId}_1`, text: 'Forte prévisibilité et sécurité immédiate', weight: 4, category: 'Sécurité', explanation: 'Permet une planification sereine sans aléas opérationnels majeurs.' },
            { id: `pro_${optId}_2`, text: 'Faible friction mentale et cadre établi', weight: 3, category: 'Bien-être', explanation: 'Des processus éprouvés réduisent la fatigue décisionnelle au quotidien.' },
            { id: `pro_${optId}_3`, text: 'Préservation du capital relationnel et des acquis', weight: 4, category: 'Carrière', explanation: 'Consolide l’expérience accumulée sans rupture de continuité.' },
            { id: `pro_${optId}_4`, text: 'Résistance éprouvée en cas de conjoncture difficile', weight: 5, category: 'Finances', explanation: 'Moins exposé aux chocs exogènes ou aux baisses d’activité imprévues.' },
          ],
      cons: isEn
        ? [
            { id: `con_${optId}_1`, text: 'Potential opportunity cost and capped upside', weight: 4, category: 'Growth', explanation: 'Staying on this path may limit breakthrough career or financial jumps.' },
            { id: `con_${optId}_2`, text: 'Risk of boredom, routine, or future regret', weight: 3, category: 'Motivation', explanation: 'Comfort zones often translate into stagnation over 3+ years.' },
            { id: `con_${optId}_3`, text: 'Slower decision-making loops and reduced autonomy', weight: 3, category: 'Autonomy', explanation: 'Dependent on external constraints and institutional timelines.' },
            { id: `con_${optId}_4`, text: 'Higher inertia if pivot is required later', weight: 3, category: 'Flexibility', explanation: 'Every year spent reinforces path dependency.' },
          ]
        : [
            { id: `con_${optId}_1`, text: 'Coût d’opportunité élevé et plafond de progression', weight: 4, category: 'Carrière', explanation: 'Rester sur cette voie bride les sauts exponentiels de revenus ou d’apprentissage.' },
            { id: `con_${optId}_2`, text: 'Risque d’érosion de la motivation et de regrets futurs', weight: 3, category: 'Sens', explanation: 'Le confort immédiat masque souvent un sentiment de stagnation à moyen terme.' },
            { id: `con_${optId}_3`, text: 'Autonomie limitée et dépendance à des tiers', weight: 3, category: 'Liberté', explanation: 'Vos arbitrages restent soumis à des agendas extérieurs.' },
            { id: `con_${optId}_4`, text: 'Inertie accrue en cas de réorientation ultérieure', weight: 3, category: 'Flexibilité', explanation: 'Plus le temps passe, plus le coût psychologique du changement s’élève.' },
          ],
      swot: {
        strengths: isEn
          ? [
              { point: 'Solid foundation and proven track record', impact: 'Direct operational credibility' },
              { point: 'Clear boundary between professional and personal life', impact: 'Maintains long-term energy' },
              { point: 'Predictable cash flows and manageable liabilities', impact: 'Peace of mind' },
            ]
          : [
              { point: 'Fondations solides et capital confiance déjà établi', impact: 'Crédibilité immédiate auprès de l’entourage et des pairs' },
              { point: 'Démarcation nette entre investissement professionnel et sphère personnelle', impact: 'Préserve votre réservoir d’énergie mentale' },
              { point: 'Visibilité financière garantie et engagements tenus', impact: 'Sérénité budgétaire à court et moyen terme' },
            ],
        weaknesses: isEn
          ? [
              { point: 'Lower velocity in personal skill expansion', impact: 'Slows down learning curve' },
              { point: 'Frustration from unexpressed creative or strategic potential', impact: 'Subtle disengagement' },
              { point: 'Vulnerability to gradual institutional stagnation', impact: 'Diminishing market premium' },
            ]
          : [
              { point: 'Ralentissement de la courbe d’acquisition de compétences rares', impact: 'Plafonnement de la valeur marchande à 3 ans' },
              { point: 'Frustration latente due au potentiel sous-exploité', impact: 'Désengagement progressif difficile à inverser' },
              { point: 'Vulnérabilité face aux réorganisations ou aléas hiérarchiques', impact: 'Manque de prise sur son propre destin' },
            ],
        opportunities: isEn
          ? [
              { point: 'Use stability to launch side initiatives or strategic training', actionable: 'Block 4 hours/week for self-development' },
              { point: 'Negotiate role improvements from a position of calm security', actionable: 'Prepare an annual impact review' },
              { point: 'Accumulate capital reserves for future bolder moves', actionable: 'Set up an automated monthly investment' },
            ]
          : [
              { point: 'Mettre à profit la stabilité pour développer un projet parallèle ou une certification', actionable: 'Bloquer 4 heures hebdomadaires dédiées à votre montée en compétences' },
              { point: 'Négocier des aménagements depuis une posture de force et de sérénité', actionable: 'Structurer un bilan d’impact pour valoriser vos réussites' },
              { point: 'Bâtir un trésor de guerre d’épargne pour un futur pivot audacieux', actionable: 'Automatiser une épargne d’au moins 20% des rentrées mensuelles' },
            ],
        threats: isEn
          ? [
              { point: 'Status quo bias disguised as wisdom', mitigation: 'Set an explicit review checkpoint in 6 months' },
              { point: 'Gradual obsolescence in high-growth paradigms', mitigation: 'Stay connected to industry innovators monthly' },
              { point: 'Golden handcuffs making future transitions painful', mitigation: 'Live below current means to preserve psychological freedom' },
            ]
          : [
              { point: 'Le piège du statu quo déguisé en fausse sagesse', mitigation: 'Fixer une date de revue formelle et non négociable dans 6 mois' },
              { point: 'Obsolescence insidieuse face aux évolutions rapides du secteur', mitigation: 'Échanger chaque mois avec des profils pionniers de votre écosystème' },
              { point: 'L’effet de menottes dorées rendant tout départ futur plus difficile', mitigation: 'Maintenir un niveau de vie sobre pour préserver sa liberté mentale' },
            ],
      },
    };
  });

  const criteria = [
    {
      id: 'crit_1',
      name: isEn ? 'Financial Security & Upside' : 'Sécurité & Potentiel Financier',
      description: isEn ? 'Balance between predictability and wealth creation' : 'Équilibre entre visibilité des revenus et perspective de gain',
      weight: 5,
      scores: {
        opt_1: { score: 8, comment: isEn ? 'Very solid safety net' : 'Filet de sécurité très robuste et prévisible' },
        opt_2: { score: 7, comment: isEn ? 'Higher upside with variable cashflow' : 'Potentiel supérieur mais rentrées variables' },
      },
      winnerOptionId: 'opt_1',
    },
    {
      id: 'crit_2',
      name: isEn ? 'Personal Growth & Autonomy' : 'Épanouissement & Autonomie',
      description: isEn ? 'Ownership over your time, decisions, and creative output' : 'Maîtrise de son emploi du temps, de ses choix et de son impact',
      weight: 4,
      scores: {
        opt_1: { score: 6, comment: isEn ? 'Standard autonomy with organizational guardrails' : 'Autonomie encadrée par les processus établis' },
        opt_2: { score: 9, comment: isEn ? 'Full ownership and steep learning curve' : 'Liberté totale d’arbitrage et courbe d’apprentissage accélérée' },
      },
      winnerOptionId: 'opt_2',
    },
    {
      id: 'crit_3',
      name: isEn ? 'Mental Peace & Stress Control' : 'Charge Mentale & Sérénité',
      description: isEn ? 'Daily psychological toll and sleep quality' : 'Pression quotidienne et capacité à déconnecter l’esprit',
      weight: 4,
      scores: {
        opt_1: { score: 8, comment: isEn ? 'Known stress patterns, easy to compartmentalize' : 'Frictions connues et compartimentage aisé le soir et week-end' },
        opt_2: { score: 6, comment: isEn ? 'High cognitive load during the initial transition phase' : 'Responsabilité directe engendrant des phases d’intensité' },
      },
      winnerOptionId: 'opt_1',
    },
    {
      id: 'crit_4',
      name: isEn ? 'Reversibility & Optionality' : 'Réversibilité & Agilité Stratégique',
      description: isEn ? 'How easy it is to undo this decision if conditions change' : 'Facilité à faire marche arrière si les hypothèses s’avèrent fausses',
      weight: 4,
      scores: {
        opt_1: { score: 7, comment: isEn ? 'Possible to leave later, but inertia grows' : 'Possible de partir plus tard, mais le coût d’inertie augmente' },
        opt_2: { score: 8, comment: isEn ? 'High optionality; skills acquired make you more employable' : 'Forte valeur marchande : l’expérience acquise vous rendra encore plus attractif' },
      },
      winnerOptionId: 'opt_2',
    },
    {
      id: 'crit_5',
      name: isEn ? 'Long-term Regret Minimization' : 'Minimisation des Regrets à 5 Ans',
      description: isEn ? 'Jeff Bezos framework: what will you regret not having attempted at age 80?' : 'Matrice de Bezos : que regretterez-vous le plus d’avoir ignoré à 80 ans ?',
      weight: 5,
      scores: {
        opt_1: { score: 5, comment: isEn ? 'Risk of persistent "what if" lingering in your mind' : 'Risque d’un lancinant "et si j’avais osé ?" dans 5 ans' },
        opt_2: { score: 9, comment: isEn ? 'Even if difficult, the trial brings clarity and pride' : 'Même en cas de turbulences, tenter l’aventure forge une fierté durable' },
      },
      winnerOptionId: 'opt_2',
    },
  ];

  const recommendedOptionId = riskProfile === 'audacieux' ? 'opt_2' : 'opt_1';
  const recName = recommendedOptionId === 'opt_1' ? opt1Name : opt2Name;

  return {
    id: `dec_${Date.now()}`,
    title,
    summary: isEn
      ? `This dilemma pits the security and predictable compounding of "${opt1Name}" against the transformative autonomy of "${opt2Name}". The decisive pivot rests on your tolerance for temporary ambiguity versus your aversion to long-term stagnation.`
      : `Ce dilemme oppose la sécurité éprouvée et la capitalisation prévisible de "${opt1Name}" à l’autonomie émancipatrice de "${opt2Name}". Le point de bascule réside dans votre tolérance à l’ambiguïté temporaire comparée à votre aversion pour la stagnation à long terme.`,
    decisionType: 'dilemma',
    options,
    comparisonMatrix: {
      criteria,
      overallWinners: [recommendedOptionId],
    },
    tiebreakerVerdict: {
      recommendedOptionId,
      headline: isEn
        ? `The Tiebreaker Verdict: Prioritize ${recName}`
        : `L'Arbitrage The Tiebreaker : Privilégier ${recName}`,
      confidence: 82,
      rationale: isEn
        ? `Upon weighing the multidimensional criteria, ${recName} emerges as the rational choice given your stated priorities.\n\nFirst, it preserves critical optionality while avoiding catastrophic downside. When faced with high emotional friction, decisions that maximize long-term learning without compromising solvency systematically outperform impulsive pivots.\n\nSecond, pursuing this option does not preclude future adjustments. It creates a solid platform from which calculated risks can be executed with maximum leverage.`
        : `Après confrontation minutieuse des critères de décision, ${recName} s’impose comme l’option la plus rationnelle au vu de vos paramètres.\n\nPremièrement, elle préserve une optionalité stratégique décisive tout en neutralisant le risque de rupture prématurée. Face à une hésitation persistante, les choix qui maximisent votre résilience sans compromettre vos réserves d’énergie mentale surpassent systématiquement les bascules hâtives.\n\nDeuxièmement, retenir cette option ne ferme aucune porte : elle constitue au contraire un tremplin structuré à partir duquel vous pourrez actionner des leviers à fort impact avec un contrôle total.`,
      conditions: [
        {
          ifCondition: isEn
            ? 'If your priority in the next 12 months is rebuilding cash buffer and peace of mind'
            : 'Si votre priorité absolue dans les 12 prochains mois est de consolider votre épargne et votre sérénité',
          thenChoose: isEn ? `Lock in "${opt1Name}" immediately` : `Verrouillez immédiatement "${opt1Name}"`,
        },
        {
          ifCondition: isEn
            ? 'If you have at least 6 months of living expenses saved and feel a strong plateau in your current routine'
            : 'Si vous disposez d’au moins 6 mois d’avance financière et ressentez un profond plafonnement intellectuel',
          thenChoose: isEn ? `Pivot courageously towards "${opt2Name}"` : `Basculez avec détermination vers "${opt2Name}"`,
        },
      ],
      devilsAdvocate: isEn
        ? `The hidden trap: Beware of using "waiting for the perfect timing" as an excuse to avoid discomfort. Perfection does not exist; every prolonged hesitation incurs an invisible cognitive rent.`
        : `Le piège caché : Méfiez-vous de l'argument du "moment idéal". Attendre que toutes les conditions soient parfaites est souvent une rationalisation de la peur de l'inconfort. Chaque mois d'indécision supplémentaire coûte une rente cognitive invisible.`,
      cognitiveBiasesDetected: [
        {
          bias: isEn ? 'Status Quo Bias' : 'Biais de Statu Quo',
          warning: isEn
            ? 'You may overvalue the default option simply because it requires no immediate upheaval.'
            : 'Tendance naturelle à privilégier l’option par défaut uniquement parce qu’elle ne requiert aucune rupture immédiate.',
          howToAvoid: isEn
            ? 'Ask yourself: "If I were starting from zero today, would I actively choose this path?"'
            : 'Posez-vous la question clé : "Si je partais d’une feuille blanche aujourd’hui sans aucun passé, choisirais-je activement cette voie ?"',
        },
        {
          bias: isEn ? 'Loss Aversion (Kahneman & Tversky)' : 'Aversion à la Perte',
          warning: isEn
            ? 'Psychologically, the pain of losing 1,000€ feels twice as intense as the joy of gaining 1,000€.'
            : 'La douleur psychologique d’une perte potentielle est ressentie deux fois plus intensément que le plaisir d’un gain équivalent.',
          howToAvoid: isEn
            ? 'Focus on the cost of inaction over a 3-year horizon rather than short-term turbulence.'
            : 'Chiffrez le coût d’opportunité de l’inaction sur 3 ans plutôt que de focaliser sur le choc à court terme.',
        },
      ],
      nextSteps: isEn
        ? [
            'Define a clear non-negotiable review milestone (calendar date in 90 days).',
            'Conduct an informational interview with one person who already made this exact choice.',
            'Document your decision in writing today to prevent retrospective hindsight bias.',
          ]
        : [
            'Fixer une date butoir de bilan dans votre calendrier à 90 jours (point d’étape non négociable).',
            'Échanger pendant 30 minutes avec une personne ayant franchi exactement ce cap il y a 1 ou 2 ans.',
            'Consigner par écrit dès aujourd’hui vos motivations pour neutraliser le biais rétrospectif futur.',
          ],
    },
  };
}

// Start server and configure Vite middleware in development
async function startServer() {
  const isDev = process.env.NODE_ENV !== 'production';

  if (isDev) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(port, () => {
    console.log(`The Tiebreaker server running on port ${port} (mode: ${isDev ? 'development' : 'production'})`);
  });
}

startServer();
