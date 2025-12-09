import React, { useEffect, useState } from 'react';
import { AlertTriangle, Info, AlertCircle, X } from 'lucide-react';

export type ModalType = 'info' | 'warning' | 'danger';

interface Props {
    isOpen: boolean;
    title: string;
    message: string;
    type?: ModalType;
    onConfirm: () => void;
    onCancel: () => void;
}

export const ConfirmationModal: React.FC<Props> = ({ isOpen, title, message, type = 'info', onConfirm, onCancel }) => {
    const [animate, setAnimate] = useState(false);

    useEffect(() => {
        if (isOpen) {
            setAnimate(true);
        } else {
            setTimeout(() => setAnimate(false), 200);
        }
    }, [isOpen]);

    if (!isOpen && !animate) return null;

    const getTheme = () => {
        switch (type) {
            case 'danger': return {
                icon: <AlertCircle className="w-6 h-6 text-red-600" />,
                bgIcon: 'bg-red-100',
                btn: 'bg-red-600 hover:bg-red-700 focus:ring-red-500',
                border: 'border-red-200'
            };
            case 'warning': return {
                icon: <AlertTriangle className="w-6 h-6 text-orange-600" />,
                bgIcon: 'bg-orange-100',
                btn: 'bg-orange-600 hover:bg-orange-700 focus:ring-orange-500',
                border: 'border-orange-200'
            };
            default: return {
                icon: <Info className="w-6 h-6 text-blue-600" />,
                bgIcon: 'bg-blue-100',
                btn: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500',
                border: 'border-blue-200'
            };
        }
    };

    const theme = getTheme();

    return (
        <div className={`fixed inset-0 z-50 flex items-center justify-center p-4 transition-opacity duration-200 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
            {/* Backdrop */}
            <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" onClick={onCancel}></div>

            {/* Modal Content */}
            <div className={`bg-white rounded-xl shadow-2xl max-w-sm w-full relative z-10 transform transition-all duration-200 scale-100 ${isOpen ? 'translate-y-0 scale-100' : 'translate-y-4 scale-95'} border ${theme.border}`}>
                <button onClick={onCancel} className="absolute top-3 right-3 text-slate-400 hover:text-slate-600">
                    <X className="w-5 h-5" />
                </button>

                <div className="p-6">
                    <div className="flex items-start gap-4">
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 ${theme.bgIcon}`}>
                            {theme.icon}
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-slate-800 leading-6 mb-2">{title}</h3>
                            <p className="text-sm text-slate-600 leading-relaxed">{message}</p>
                        </div>
                    </div>

                    <div className="mt-8 flex justify-end gap-3">
                        <button
                            onClick={onCancel}
                            className="px-4 py-2 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-100 border border-transparent transition-colors"
                        >
                            取消
                        </button>
                        <button
                            onClick={onConfirm}
                            className={`px-4 py-2 rounded-lg text-sm font-bold text-white shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all ${theme.btn}`}
                        >
                            确认执行
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
