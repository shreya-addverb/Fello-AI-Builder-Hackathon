import {motion} from 'framer-motion';
import {AlertCircle,Clock3,Database,ExternalLink,MinusCircle} from 'lucide-react';
import {label,percent} from '../lib/utils';
import {Expandable,JsonViewer,StageIcon} from './ui';
import type {AnalysisContext,Status} from '../types';

export const stages=[
  ['company_identification','Company Identification','CompanyIdentifier'],
  ['company_enrichment','Company Enrichment','CompanyEnrichment'],
  ['technology_stack','Technology Detection','TechnologyDetection'],
  ['leadership','Leadership Discovery','LeadershipDiscovery'],
  ['business_signals','Business Signals','BusinessSignals'],
  ['persona','Persona Inference','PersonaInference'],
  ['intent','Intent Scoring','IntentScoring'],
  ['ai_summary','AI Summary','SummaryGeneration'],
  ['sales_recommendations','Recommendations','SalesRecommendationGeneration'],
] as const;

export function PipelineList({status,completed=[],context}:{status:Status;completed?:string[];context?:AnalysisContext}){
  const outcomes=context?.pipeline_metadata.stage_results||[];
  return <div className="pipeline-list">{stages.map(([key,name,className],index)=>{
    const outcome=outcomes.find(item=>item.stage===className);
    const done=completed.includes(className)||status==='completed';
    const current=status==='running'&&!outcome&&index===completed.length;
    const stageStatus=outcome?.status||(current?'running':done?'completed':'idle');
    const data=context?.[key as keyof AnalysisContext];
    return <motion.div initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} transition={{delay:index*.04}} className="stage" key={key}>
      <div className="stage-line">{stageStatus==='no_data'||stageStatus==='skipped'?<MinusCircle className="muted"/>:stageStatus==='failed'?<AlertCircle className="red"/>:<StageIcon status={stageStatus}/>} {index<stages.length-1&&<i/>}</div>
      <div className="stage-content">
        <div className="stage-title"><div><span>{name}</span><small>Stage {String(index+1).padStart(2,'0')}</small></div><span className="stage-state">
          {stageStatus==='running'&&<><Clock3/> Processing</>}{stageStatus==='completed'&&'Complete'}{stageStatus==='no_data'&&'No evidence'}{stageStatus==='skipped'&&'Not applicable'}{stageStatus==='failed'&&'Failed'}{stageStatus==='idle'&&'Waiting'}
        </span></div>
        {outcome?.message&&<p className="stage-message">{outcome.message}</p>}
        {data&&<Expandable title="Structured output" meta={<><Database size={14}/>{Object.keys(data as object).length} fields</>}><JsonViewer data={data}/></Expandable>}
      </div>
    </motion.div>
  })}</div>
}

export function TechnologyGrid({stack}:{stack:AnalysisContext['technology_stack']}){
  const groups=Object.entries(stack).filter(([,value])=>Array.isArray(value)&&value.length) as [string,Array<{name:string;confidence:number}>][];
  return groups.length?<div className="tech-groups">{groups.map(([group,items])=><div key={group}><p className="eyebrow">{label(group)}</p><div className="chips">{items.map(item=><span className="tech-chip" key={item.name}>{item.name}<small>{percent(item.confidence)}</small></span>)}</div></div>)}</div>:<p className="muted">No directly supported technologies were detected.</p>
}

export function SourceLink({href}:{href:string}){return <a href={href} target="_blank" rel="noreferrer" className="source-link">Source <ExternalLink size={12}/></a>}
