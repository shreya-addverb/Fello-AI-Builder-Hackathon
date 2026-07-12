export const cn=(...values:Array<string|false|null|undefined>)=>values.filter(Boolean).join(' ');
export const percent=(value?:number|null)=>value==null?'—':`${Math.round(value*100)}%`;
export const label=(value:string)=>value.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
