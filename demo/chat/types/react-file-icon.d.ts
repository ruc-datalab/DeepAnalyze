declare module "react-file-icon" {
  import * as React from "react";
  export const defaultStyles: Record<string, any>;
  export interface FileIconProps {
    extension?: string;
    [key: string]: any;
  }
  export const FileIcon: React.FC<FileIconProps>;
}
