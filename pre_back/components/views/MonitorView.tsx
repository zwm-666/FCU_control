import * as React from 'react';
import { MachineState } from '../../types';
import { IndustrialSchematic } from '../IndustrialSchematic';

interface Props {
  data: MachineState;
}

export const MonitorView: React.FC<Props> = ({ data }) => (
  <div className="w-full h-full p-2 flex flex-col items-center bg-transparent pt-4 overflow-hidden">
    <div className="w-full flex-1 flex justify-center items-start overflow-hidden">
      <div className="scale-100 origin-top w-full h-full">
        <IndustrialSchematic data={data} />
      </div>
    </div>
  </div>
);
