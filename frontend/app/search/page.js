import { redirect } from "next/navigation";

// Search now lives on the landing page.
export default function SearchRedirect() {
  redirect("/");
}
